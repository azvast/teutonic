"""Dataset infrastructure: manifest-driven shards on R2 with range-request support.

The v1 dataset is stored as flat 1D numpy arrays of uint32 token IDs on R2,
organized under dataset/v1/ with a manifest.json that catalogs all shards.

Supports:
- Manifest loading/caching for shard discovery
- Range requests to fetch individual sequences without downloading full shards
- Local shard loading via memory-mapped .npy files
- Deterministic shard + index selection from block hash
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import struct
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

DTYPE = np.dtype("<u4")
BYTES_PER_TOKEN = DTYPE.itemsize  # 4
MANIFEST_KEY = "dataset/v1/manifest.json"
NPY_HEADER_MAX = 1024  # max bytes to fetch for npy header parsing


class DatasetManifest:
    """Parsed v1 dataset manifest from R2."""

    def __init__(self, data: dict):
        self.version = data["version"]
        self.tokenizer = data.get("tokenizer", "")
        self.dtype = data.get("dtype", "uint32")
        self.total_tokens = data["total_tokens"]
        self.total_shards = data["total_shards"]
        self.shard_prefix = data.get("shard_prefix", "dataset/v1/shards/")
        self.shards = data["shards"]

    def shard_key(self, idx: int) -> str:
        return self.shards[idx]["key"]

    def shard_n_tokens(self, idx: int) -> int:
        return self.shards[idx]["n_tokens"]

    def shard_sha256(self, idx: int) -> str:
        return self.shards[idx].get("sha256", "")

    @classmethod
    def from_json(cls, raw: str | bytes | dict) -> DatasetManifest:
        if isinstance(raw, (str, bytes)):
            raw = json.loads(raw)
        return cls(raw)


def load_manifest(r2_client) -> DatasetManifest:
    """Fetch the v1 manifest from R2."""
    data = r2_client.get_json(MANIFEST_KEY)
    if data is None:
        raise FileNotFoundError(f"Manifest not found at {MANIFEST_KEY}")
    return DatasetManifest.from_json(data)


def load_manifest_cached(r2_client, cache_dir: str | Path = "/tmp/teutonic") -> DatasetManifest:
    """Load manifest from local cache, falling back to R2."""
    cache_path = Path(cache_dir) / "manifest.json"
    if cache_path.exists():
        logger.debug("Loading manifest from cache: %s", cache_path)
        return DatasetManifest.from_json(json.loads(cache_path.read_text()))

    manifest = load_manifest(r2_client)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({
        "version": manifest.version,
        "tokenizer": manifest.tokenizer,
        "dtype": manifest.dtype,
        "total_tokens": manifest.total_tokens,
        "total_shards": manifest.total_shards,
        "shard_prefix": manifest.shard_prefix,
        "shards": manifest.shards,
    }, indent=2))
    logger.info("Manifest cached to %s (%d shards)", cache_path, manifest.total_shards)
    return manifest


def _parse_npy_header(header_bytes: bytes) -> tuple[tuple, np.dtype, int]:
    """Parse a .npy header from raw bytes. Returns (shape, dtype, data_offset)."""
    buf = io.BytesIO(header_bytes)
    magic = buf.read(6)
    if magic != b"\x93NUMPY":
        raise ValueError("Not a valid .npy file header")

    version = struct.unpack("BB", buf.read(2))
    if version[0] == 1:
        header_len = struct.unpack("<H", buf.read(2))[0]
    else:
        header_len = struct.unpack("<I", buf.read(4))[0]

    header_str = buf.read(header_len).decode("latin1").strip()
    data_offset = buf.tell()

    header_dict = eval(header_str)  # noqa: S307 — numpy header format
    shape = tuple(header_dict["shape"])
    dtype = np.dtype(header_dict["descr"])

    return shape, dtype, data_offset


def select_eval_shard(
    n_shards: int,
    commit_block_hash: str,
    hotkey: str,
) -> int:
    """Deterministically select a shard index from the manifest.

    Uses the same seed material as sequence index selection so the shard
    choice is unpredictable until the commit block is finalized.
    """
    seed_material = f"{commit_block_hash}:{hotkey}".encode()
    seed_hash = hashlib.blake2b(seed_material, digest_size=8).digest()
    seed = int.from_bytes(seed_hash, "little")
    rng = np.random.Generator(np.random.PCG64(seed))
    return int(rng.integers(0, n_shards))


def select_eval_indices(
    n_sequences: int,
    N: int,
    commit_block_hash: str,
    hotkey: str,
) -> list[int]:
    """Deterministically select N eval sequence indices.

    Uses blake2b(commit_block_hash || hotkey) as the random seed so eval
    data is unpredictable until the commit block is finalized.
    """
    seed_material = f"{commit_block_hash}:{hotkey}".encode()
    seed_hash = hashlib.blake2b(seed_material, digest_size=8).digest()
    seed = int.from_bytes(seed_hash, "little")

    rng = np.random.Generator(np.random.PCG64(seed))

    if N >= n_sequences:
        indices = list(range(n_sequences))
        rng.shuffle(indices)
        return indices

    return rng.choice(n_sequences, size=N, replace=False).tolist()


def fetch_sequences_by_range(
    s3_client,
    bucket: str,
    shard_key: str,
    seq_indices: list[int],
    seq_len: int,
) -> dict[int, torch.Tensor]:
    """Fetch specific sequences from a shard via S3 range requests.

    Returns a dict mapping sequence index -> token tensor (int64 for torch).
    Groups nearby indices into contiguous range requests to minimize HTTP calls.
    """
    # First, read the npy header to find data offset
    header_resp = s3_client.get_object(
        Bucket=bucket, Key=shard_key, Range=f"bytes=0-{NPY_HEADER_MAX - 1}"
    )
    header_bytes = header_resp["Body"].read()
    _shape, _dtype, data_offset = _parse_npy_header(header_bytes)

    tokens_per_seq = seq_len
    bytes_per_seq = tokens_per_seq * BYTES_PER_TOKEN

    # Sort indices and group into contiguous runs to batch range requests.
    # Two sequences within GAP_THRESHOLD of each other get merged into one request.
    GAP_THRESHOLD = 64  # merge if gap is ≤ 64 sequences
    sorted_indices = sorted(set(seq_indices))

    groups: list[tuple[int, int]] = []  # (start_idx, end_idx) inclusive
    if sorted_indices:
        g_start = g_end = sorted_indices[0]
        for idx in sorted_indices[1:]:
            if idx - g_end <= GAP_THRESHOLD:
                g_end = idx
            else:
                groups.append((g_start, g_end))
                g_start = g_end = idx
        groups.append((g_start, g_end))

    results: dict[int, torch.Tensor] = {}
    needed = set(seq_indices)

    for g_start, g_end in groups:
        byte_start = data_offset + g_start * bytes_per_seq
        byte_end = data_offset + (g_end + 1) * bytes_per_seq - 1

        resp = s3_client.get_object(
            Bucket=bucket, Key=shard_key, Range=f"bytes={byte_start}-{byte_end}"
        )
        chunk = resp["Body"].read()

        for idx in range(g_start, g_end + 1):
            if idx not in needed:
                continue
            local_offset = (idx - g_start) * bytes_per_seq
            token_bytes = chunk[local_offset : local_offset + bytes_per_seq]
            arr = np.frombuffer(token_bytes, dtype=DTYPE).copy()
            results[idx] = torch.from_numpy(arr).to(torch.long)

    return results


class EvalDataset(Dataset):
    """Memory-mapped evaluation dataset from pre-tokenized .npy shards.

    Each item is a sequence of `seq_len` token IDs sliced from the flat array.
    """

    def __init__(self, shard_path: str | Path, seq_len: int):
        self.seq_len = seq_len
        shard_path = Path(shard_path)

        if not shard_path.exists():
            raise FileNotFoundError(f"Shard not found: {shard_path}")

        arr = np.load(str(shard_path), mmap_mode="r", allow_pickle=False)
        if arr.dtype != np.uint32:
            arr = arr.astype(np.uint32, copy=False)
        if arr.ndim != 1:
            arr = arr.reshape(-1)

        self.tokens = torch.from_numpy(arr)
        total_tokens = self.tokens.shape[0]

        usable = (total_tokens // seq_len) * seq_len
        self.tokens = self.tokens[:usable]
        self.n_sequences = usable // seq_len

        logger.info(
            "EvalDataset: %s, %d tokens, %d sequences of length %d",
            shard_path.name, usable, self.n_sequences, seq_len,
        )

    def __len__(self) -> int:
        return self.n_sequences

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx < 0 or idx >= self.n_sequences:
            raise IndexError(f"Index {idx} out of range [0, {self.n_sequences})")
        start = idx * self.seq_len
        return self.tokens[start : start + self.seq_len]


def download_shard_from_r2(
    r2_client,
    shard_key: str,
    local_path: str | Path,
) -> Path:
    """Download a dataset shard from R2 to local disk."""
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if local_path.exists():
        logger.info("Shard already cached at %s", local_path)
        return local_path

    logger.info("Downloading shard %s -> %s", shard_key, local_path)
    r2_client.download_file(shard_key, str(local_path))
    logger.info("Downloaded shard: %.2f GB", local_path.stat().st_size / 1e9)
    return local_path
