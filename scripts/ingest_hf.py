#!/usr/bin/env python3
"""Stream a HuggingFace dataset, tokenize, pack into .npy shards, and upload
to Hippius. Maintains a live manifest so validators can use shards as soon as
they appear.

Supports any parquet-based HF dataset with a 'text' column. Tested with:
  - uonlp/CulturaX (6.3T tokens, 167 languages, English-first)
  - nvidia/Nemotron-CC-v2 (10.3TB, cleaned Common Crawl)

Usage:
    python scripts/ingest_hf.py --dataset uonlp/CulturaX [--langs en,de,fr] [--dry-run]
    python scripts/ingest_hf.py --dataset nvidia/Nemotron-CC-v2 [--dry-run]

Env vars:
    HF_TOKEN                    HuggingFace token (gated datasets)
    TEUTONIC_DS_ENDPOINT        Hippius S3 endpoint (default: https://s3.hippius.com)
    TEUTONIC_DS_BUCKET          Hippius bucket (default: teutonic-sn3)
    TEUTONIC_DS_ACCESS_KEY      Hippius access key
    TEUTONIC_DS_SECRET_KEY      Hippius secret key
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path

import boto3
import numpy as np
import pyarrow.parquet as pq
from botocore.config import Config as BotoConfig
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoTokenizer

log = logging.getLogger("ingest_hf")

TOKENIZER_NAME = "unsloth/gemma-3-1b-it"
DEST_PREFIX = "dataset/v2"
DTYPE = np.dtype("<u4")
BYTES_PER_TOKEN = DTYPE.itemsize  # 4

DS_ENDPOINT = os.environ.get("TEUTONIC_DS_ENDPOINT", "https://s3.hippius.com")
DS_BUCKET = os.environ.get("TEUTONIC_DS_BUCKET", "teutonic-sn3")
DS_ACCESS_KEY = os.environ.get("TEUTONIC_DS_ACCESS_KEY", "")
DS_SECRET_KEY = os.environ.get("TEUTONIC_DS_SECRET_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

_shutdown = False


def _handle_signal(signum, frame):
    global _shutdown
    log.warning("received signal %s, will stop after current shard", signum)
    _shutdown = True


# ---------------------------------------------------------------------------
# Hippius S3
# ---------------------------------------------------------------------------

def make_client():
    return boto3.client(
        "s3",
        endpoint_url=DS_ENDPOINT,
        aws_access_key_id=DS_ACCESS_KEY,
        aws_secret_access_key=DS_SECRET_KEY,
        region_name="decentralized",
        config=BotoConfig(
            signature_version="s3v4",
            retries={"max_attempts": 3, "mode": "adaptive"},
            s3={"addressing_style": "path"},
        ),
    )


def get_manifest(client) -> dict | None:
    try:
        body = client.get_object(
            Bucket=DS_BUCKET, Key=f"{DEST_PREFIX}/manifest.json"
        )["Body"].read()
        return json.loads(body)
    except Exception:
        return None


def put_manifest(client, manifest: dict):
    body = json.dumps(manifest, indent=2).encode()
    client.put_object(
        Bucket=DS_BUCKET,
        Key=f"{DEST_PREFIX}/manifest.json",
        Body=body,
        ContentType="application/json",
    )


def flush_shard(buf: bytearray, shard_idx: int, client, dry_run: bool) -> dict:
    """Write token buffer as .npy shard, upload, return shard info dict."""
    key = f"{DEST_PREFIX}/shards/shard_{shard_idx:06d}.npy"
    n_tokens = len(buf) // BYTES_PER_TOKEN

    arr = np.frombuffer(buf, dtype=DTYPE)
    tmp_path = f"/tmp/ingest_shard_{shard_idx:06d}.npy"
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

    if not dry_run:
        client.upload_file(tmp_path, DS_BUCKET, key)
        log.info("uploaded %s (%.2f GB, %s tokens, sha256=%s...)",
                 key, size_bytes / 1e9, f"{n_tokens:,}", sha[:16])
    else:
        log.info("[DRY RUN] would upload %s (%.2f GB, %s tokens)",
                 key, size_bytes / 1e9, f"{n_tokens:,}")

    Path(tmp_path).unlink(missing_ok=True)

    return {
        "key": key,
        "n_tokens": n_tokens,
        "size_bytes": size_bytes,
        "sha256": sha,
    }


def update_manifest(client, shard_info: dict, manifest: dict | None,
                    dataset_repo: str, dry_run: bool) -> dict:
    """Append shard to manifest and upload atomically."""
    if manifest is None:
        manifest = {
            "version": "v2",
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "tokenizer": TOKENIZER_NAME,
            "dtype": "uint32",
            "source": dataset_repo,
            "total_tokens": 0,
            "total_shards": 0,
            "shard_prefix": f"{DEST_PREFIX}/shards/",
            "shards": [],
        }

    manifest["shards"].append(shard_info)
    manifest["total_shards"] = len(manifest["shards"])
    manifest["total_tokens"] = sum(s["n_tokens"] for s in manifest["shards"])
    manifest["updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    if not dry_run:
        put_manifest(client, manifest)
        log.info("manifest updated: %d shards, %s total tokens",
                 manifest["total_shards"], f"{manifest['total_tokens']:,}")
    else:
        log.info("[DRY RUN] manifest would have %d shards, %s total tokens",
                 manifest["total_shards"], f"{manifest['total_tokens']:,}")

    return manifest


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def tokenize_and_pack(tokenizer, text: str, remainder: list[int],
                      seq_len: int) -> tuple[bytes, list[int]]:
    """Tokenize text, prepend leftover tokens from previous sample, and pack
    into full seq_len windows. Returns (packed_bytes, new_remainder)."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    all_tokens = remainder + ids

    n_full = len(all_tokens) // seq_len
    packed_count = n_full * seq_len

    if packed_count == 0:
        return b"", all_tokens

    arr = np.array(all_tokens[:packed_count], dtype=DTYPE)
    return arr.tobytes(), all_tokens[packed_count:]


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------

CULTURAX_LANG_PRIORITY = [
    "en", "de", "fr", "es", "it", "pt", "nl", "ru", "zh", "ja", "ko",
    "pl", "cs", "sv", "da", "no", "fi", "ro", "hu", "el", "bg", "tr",
]

NEMOTRON_CONFIG_PRIORITY = [
    "High-Quality", "Medium-High-Quality", "Medium-Quality",
    "Diverse-QA", "High-Quality-Synthetic", "Translated-Diverse-QA",
]


def discover_parquet_files(dataset_repo: str, token: str,
                           langs: list[str] | None = None) -> list[tuple[str, str]]:
    """Return [(config_name, file_path), ...] ordered by priority."""
    api = HfApi(token=token)
    all_files = api.list_repo_files(dataset_repo, repo_type="dataset")

    config_files: dict[str, list[str]] = {}
    for f in all_files:
        if f.endswith(".parquet") and "/" in f:
            config = f.split("/")[0]
            config_files.setdefault(config, []).append(f)

    for config in config_files:
        config_files[config].sort()

    is_culturax = "CulturaX" in dataset_repo
    if langs:
        ordered_configs = [l for l in langs if l in config_files]
    elif is_culturax:
        ordered_configs = [l for l in CULTURAX_LANG_PRIORITY if l in config_files]
        for c in sorted(config_files):
            if c not in ordered_configs:
                ordered_configs.append(c)
    else:
        ordered_configs = [p for p in NEMOTRON_CONFIG_PRIORITY if p in config_files]
        for c in sorted(config_files):
            if c not in ordered_configs:
                ordered_configs.append(c)

    result = []
    for config in ordered_configs:
        files = config_files[config]
        log.info("config %s: %d parquet files", config, len(files))
        for f in files:
            result.append((config, f))

    log.info("total: %d parquet files across %d configs", len(result), len(ordered_configs))
    return result


def iter_parquet_texts(parquet_path: str):
    """Yield text strings from a local parquet file, row-group by row-group."""
    pf = pq.ParquetFile(parquet_path)
    for rg_idx in range(pf.metadata.num_row_groups):
        table = pf.read_row_group(rg_idx, columns=["text"])
        texts = table.column("text").to_pylist()
        for text in texts:
            if text:
                yield text


# ---------------------------------------------------------------------------
# Main ingestion loop
# ---------------------------------------------------------------------------

def ingest(dataset_repo: str, shard_size_gb: float = 2.0, seq_len: int = 2048,
           dry_run: bool = False, langs: list[str] | None = None):
    global _shutdown
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    shard_size_bytes = int(shard_size_gb * 1024**3)
    shard_size_bytes = (shard_size_bytes // BYTES_PER_TOKEN) * BYTES_PER_TOKEN

    for var in ["TEUTONIC_DS_ACCESS_KEY", "TEUTONIC_DS_SECRET_KEY"]:
        if not os.environ.get(var):
            log.error("missing env var: %s", var)
            sys.exit(1)

    token = HF_TOKEN
    if not token:
        log.error("missing HF_TOKEN for gated dataset")
        sys.exit(1)

    client = make_client()

    log.info("loading tokenizer %s", TOKENIZER_NAME)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, token=token)
    log.info("tokenizer loaded: vocab_size=%d", tokenizer.vocab_size)

    manifest = get_manifest(client)
    if manifest:
        start_shard_idx = manifest["total_shards"]
        log.info("resuming: %d existing shards, %s tokens already ingested",
                 start_shard_idx, f"{manifest['total_tokens']:,}")
    else:
        start_shard_idx = 0
        manifest = None
        log.info("starting fresh (no existing manifest)")

    log.info("dataset: %s", dataset_repo)
    log.info("discovering parquet files...")
    parquet_files = discover_parquet_files(dataset_repo, token, langs)

    state_path = Path(f"/tmp/ingest_state_{dataset_repo.replace('/', '_')}.json")
    completed_files: set[str] = set()
    if state_path.exists() and start_shard_idx > 0:
        try:
            state = json.loads(state_path.read_text())
            completed_files = set(state.get("completed_files", []))
            log.info("resume state: %d files already processed", len(completed_files))
        except Exception:
            pass

    shard_idx = start_shard_idx
    buf = bytearray()
    remainder: list[int] = []
    total_samples = 0
    total_tokens_ingested = 0
    t0 = time.time()

    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 10

    for config_name, file_path in parquet_files:
        if _shutdown:
            break

        if file_path in completed_files:
            consecutive_failures = 0
            continue

        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            log.error("aborting: %d consecutive download failures (dataset access may be pending)",
                       consecutive_failures)
            break

        log.info("downloading %s", file_path)
        try:
            local_path = hf_hub_download(
                dataset_repo,
                file_path,
                repo_type="dataset",
                token=token,
            )
            consecutive_failures = 0
        except Exception as e:
            log.warning("failed to download %s: %s", file_path, e)
            consecutive_failures += 1
            continue

        log.info("processing %s", file_path)
        file_samples = 0

        for text in iter_parquet_texts(local_path):
            if _shutdown:
                break

            packed, remainder = tokenize_and_pack(tokenizer, text, remainder, seq_len)
            if not packed:
                continue

            packed_tokens = len(packed) // BYTES_PER_TOKEN
            buf.extend(packed)
            total_samples += 1
            file_samples += 1
            total_tokens_ingested += packed_tokens

            if len(buf) >= shard_size_bytes:
                shard_data = bytes(buf[:shard_size_bytes])
                buf = bytearray(buf[shard_size_bytes:])

                shard_info = flush_shard(
                    bytearray(shard_data), shard_idx, client, dry_run,
                )
                manifest = update_manifest(
                    client, shard_info, manifest, dataset_repo, dry_run,
                )
                shard_idx += 1

                elapsed = time.time() - t0
                rate = total_tokens_ingested / elapsed if elapsed > 0 else 0
                log.info(
                    "progress: shard=%d samples=%s tokens=%s rate=%.0f tok/s elapsed=%.0fs",
                    shard_idx, f"{total_samples:,}", f"{total_tokens_ingested:,}",
                    rate, elapsed,
                )

            if total_samples % 50_000 == 0 and total_samples > 0:
                elapsed = time.time() - t0
                rate = total_tokens_ingested / elapsed if elapsed > 0 else 0
                log.info(
                    "heartbeat: samples=%s tokens=%s buf=%.1f MB rate=%.0f tok/s",
                    f"{total_samples:,}", f"{total_tokens_ingested:,}",
                    len(buf) / 1e6, rate,
                )

        completed_files.add(file_path)
        state_path.write_text(json.dumps({
            "completed_files": sorted(completed_files),
            "shard_idx": shard_idx,
            "total_samples": total_samples,
            "total_tokens": total_tokens_ingested,
        }))

        log.info("finished %s: %d samples from this file (total: %s)",
                 file_path, file_samples, f"{total_samples:,}")

    if buf and not _shutdown:
        shard_info = flush_shard(buf, shard_idx, client, dry_run)
        manifest = update_manifest(client, shard_info, manifest, dataset_repo, dry_run)
        shard_idx += 1

    elapsed = time.time() - t0
    log.info(
        "ingestion %s: %d shards, %s samples, %s tokens in %.0fs",
        "stopped (signal)" if _shutdown else "complete",
        shard_idx - start_shard_idx,
        f"{total_samples:,}",
        f"{total_tokens_ingested:,}",
        elapsed,
    )


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Ingest HuggingFace dataset to Hippius shards")
    parser.add_argument("--dataset", default="uonlp/CulturaX",
                        help="HF dataset repo (default: uonlp/CulturaX)")
    parser.add_argument("--langs", default=None,
                        help="Comma-separated language codes for CulturaX (default: en first, then others)")
    parser.add_argument("--shard-size-gb", type=float, default=2.0)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    langs = args.langs.split(",") if args.langs else None

    ingest(
        dataset_repo=args.dataset,
        shard_size_gb=args.shard_size_gb,
        seq_len=args.seq_len,
        dry_run=args.dry_run,
        langs=langs,
    )


if __name__ == "__main__":
    main()
