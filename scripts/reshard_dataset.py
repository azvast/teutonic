#!/usr/bin/env python3
"""Reshard the legacy 2D anneal .npy files into smaller 1D flat shards with a manifest.

Streams each legacy file from R2 via range requests, flattens to 1D uint32,
writes ~2 GB chunks as new shards, computes SHA256 per shard, and builds
a manifest.json. Uploads everything to dataset/v1/ on the destination bucket.

Usage:
    python scripts/reshard_dataset.py [--dry-run] [--shard-size-gb 2]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import boto3
import numpy as np
from botocore.config import Config as BotoConfig

DTYPE = np.dtype("<u4")
BYTES_PER_TOKEN = DTYPE.itemsize  # 4

R2_CFG = {
    "endpoint": "https://00523074f51300584834607253cae0fa.r2.cloudflarestorage.com",
    "access_key": "f0e13641fae0ecedc476f81a4aca6903",
    "secret_key": "df530458cb884d3365a59abb5aa5d72b02576132c44fac51a6e4dddf7548742e",
    "bucket": "constantinople",
}

LEGACY_SHARDS = [
    "anneal/anneal_000000.npy",
    "anneal/anneal_000002.npy",
    "anneal/anneal_000004.npy",
    "anneal/anneal_000005.npy",
]

DEST_PREFIX = "dataset/v1"
DOWNLOAD_CHUNK = 128 * 1024 * 1024  # 128 MB per range request


def make_client():
    return boto3.client(
        "s3",
        endpoint_url=R2_CFG["endpoint"],
        aws_access_key_id=R2_CFG["access_key"],
        aws_secret_access_key=R2_CFG["secret_key"],
        region_name="auto",
        config=BotoConfig(
            connect_timeout=30,
            read_timeout=120,
            retries={"max_attempts": 3, "mode": "adaptive"},
        ),
    )


def get_object_size(client, key: str) -> int:
    resp = client.head_object(Bucket=R2_CFG["bucket"], Key=key)
    return resp["ContentLength"]


def read_npy_header(client, key: str) -> tuple[tuple, np.dtype, int]:
    """Read a .npy file header via range request. Returns (shape, dtype, header_bytes)."""
    header_data = client.get_object(
        Bucket=R2_CFG["bucket"], Key=key, Range="bytes=0-1023"
    )["Body"].read()

    buf = io.BytesIO(header_data)
    magic = buf.read(6)
    if magic != b"\x93NUMPY":
        raise ValueError(f"Not a valid .npy file: {key}")

    version = struct.unpack("BB", buf.read(2))
    if version[0] == 1:
        header_len = struct.unpack("<H", buf.read(2))[0]
    else:
        header_len = struct.unpack("<I", buf.read(4))[0]

    header_str = buf.read(header_len).decode("latin1").strip()
    header_offset = buf.tell()

    header_dict = eval(header_str)  # noqa: S307 — numpy header is trusted
    shape = tuple(header_dict["shape"])
    dtype = np.dtype(header_dict["descr"])

    return shape, dtype, header_offset


@dataclass
class ShardInfo:
    key: str
    n_tokens: int
    size_bytes: int
    sha256: str


@dataclass
class ReshardState:
    shard_size_bytes: int
    shard_idx: int = 0
    shards: list[ShardInfo] = field(default_factory=list)
    total_tokens: int = 0

    # Current shard accumulation buffer
    _buf: bytearray = field(default_factory=bytearray, repr=False)
    _hasher: hashlib._Hash = field(default_factory=lambda: hashlib.sha256(), repr=False)
    _tokens_in_buf: int = 0

    def flush_shard(self, client, dry_run: bool) -> ShardInfo | None:
        if not self._buf:
            return None

        key = f"{DEST_PREFIX}/shards/shard_{self.shard_idx:06d}.npy"
        n_tokens = self._tokens_in_buf

        arr = np.frombuffer(self._buf, dtype=DTYPE)
        tmp_path = f"/tmp/teutonic_reshard_{self.shard_idx:06d}.npy"
        np.save(tmp_path, arr)

        file_hasher = hashlib.sha256()
        with open(tmp_path, "rb") as f:
            while True:
                chunk = f.read(8 * 1024 * 1024)
                if not chunk:
                    break
                file_hasher.update(chunk)

        size_bytes = Path(tmp_path).stat().st_size
        sha = file_hasher.hexdigest()
        info = ShardInfo(key=key, n_tokens=n_tokens, size_bytes=size_bytes, sha256=sha)

        if not dry_run:
            client.upload_file(tmp_path, R2_CFG["bucket"], key)
            print(f"  Uploaded {key} ({size_bytes / 1e9:.2f} GB, {n_tokens:,} tokens)")
        else:
            print(f"  [DRY RUN] Would upload {key} ({size_bytes / 1e9:.2f} GB, {n_tokens:,} tokens)")

        Path(tmp_path).unlink(missing_ok=True)
        self.shards.append(info)
        self.total_tokens += n_tokens
        self.shard_idx += 1

        self._buf = bytearray()
        self._hasher = hashlib.sha256()
        self._tokens_in_buf = 0

        return info

    def add_data(self, data: bytes, client, dry_run: bool):
        """Add raw token bytes, flushing shards as needed."""
        offset = 0
        while offset < len(data):
            remaining_capacity = self.shard_size_bytes - len(self._buf)
            take = min(remaining_capacity, len(data) - offset)
            self._buf.extend(data[offset : offset + take])
            self._tokens_in_buf += take // BYTES_PER_TOKEN
            offset += take

            if len(self._buf) >= self.shard_size_bytes:
                self.flush_shard(client, dry_run)


def reshard(shard_size_gb: float = 2.0, dry_run: bool = False):
    shard_size_bytes = int(shard_size_gb * 1024**3)
    # Align to token boundary
    shard_size_bytes = (shard_size_bytes // BYTES_PER_TOKEN) * BYTES_PER_TOKEN

    client = make_client()
    state = ReshardState(shard_size_bytes=shard_size_bytes)

    print(f"Resharding {len(LEGACY_SHARDS)} legacy files -> ~{shard_size_gb} GB shards")
    print(f"Destination: {R2_CFG['bucket']}/{DEST_PREFIX}/")
    if dry_run:
        print("DRY RUN — nothing will be uploaded\n")

    t0 = time.time()
    total_source_bytes = 0

    for legacy_key in LEGACY_SHARDS:
        print(f"\nProcessing {legacy_key}...")

        file_size = get_object_size(client, legacy_key)
        shape, dtype, header_offset = read_npy_header(client, legacy_key)
        data_bytes = file_size - header_offset

        if dtype != DTYPE:
            print(f"  WARNING: dtype is {dtype}, expected {DTYPE}. Will convert.")

        total_elements = 1
        for s in shape:
            total_elements *= s
        print(f"  Shape: {shape}, dtype: {dtype}, data: {data_bytes / 1e9:.1f} GB, elements: {total_elements:,}")

        # Stream the data portion via range requests
        offset = header_offset
        downloaded = 0
        while offset < file_size:
            end = min(offset + DOWNLOAD_CHUNK - 1, file_size - 1)
            resp = client.get_object(
                Bucket=R2_CFG["bucket"], Key=legacy_key,
                Range=f"bytes={offset}-{end}",
            )
            chunk = resp["Body"].read()

            # Ensure we only pass complete tokens
            usable = (len(chunk) // BYTES_PER_TOKEN) * BYTES_PER_TOKEN
            if usable < len(chunk):
                # Adjust offset to re-read the partial token next iteration
                end = offset + usable - 1

            state.add_data(chunk[:usable], client, dry_run)
            downloaded += usable
            offset = end + 1

            pct = downloaded / data_bytes * 100
            print(f"  {downloaded / 1e9:.1f} / {data_bytes / 1e9:.1f} GB ({pct:.0f}%) — {state.shard_idx} shards so far", end="\r")

        total_source_bytes += data_bytes
        print(f"\n  Done with {legacy_key}: {downloaded / 1e9:.1f} GB processed")

    # Flush any remaining data in the buffer
    state.flush_shard(client, dry_run)

    elapsed = time.time() - t0

    # Build manifest
    manifest = {
        "version": "v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tokenizer": "google/gemma-3-1b",
        "dtype": "uint32",
        "total_tokens": state.total_tokens,
        "total_shards": len(state.shards),
        "shard_prefix": f"{DEST_PREFIX}/shards/",
        "shards": [
            {
                "key": s.key,
                "n_tokens": s.n_tokens,
                "size_bytes": s.size_bytes,
                "sha256": s.sha256,
            }
            for s in state.shards
        ],
    }

    manifest_key = f"{DEST_PREFIX}/manifest.json"
    if not dry_run:
        body = json.dumps(manifest, indent=2).encode()
        client.put_object(Bucket=R2_CFG["bucket"], Key=manifest_key, Body=body, ContentType="application/json")
        print(f"\nManifest uploaded to {manifest_key}")
    else:
        print(f"\n[DRY RUN] Would upload manifest to {manifest_key}")

    print(f"\nResharding complete:")
    print(f"  Source: {len(LEGACY_SHARDS)} files, {total_source_bytes / 1e12:.2f} TB")
    print(f"  Output: {len(state.shards)} shards, {state.total_tokens:,} tokens")
    print(f"  Time: {elapsed / 3600:.1f} hours ({elapsed:.0f}s)")

    # Save manifest locally too
    manifest_path = Path("manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"  Local manifest: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Reshard legacy dataset into v1 format")
    parser.add_argument("--shard-size-gb", type=float, default=2.0, help="Target shard size in GB")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without uploading")
    args = parser.parse_args()

    reshard(shard_size_gb=args.shard_size_gb, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
