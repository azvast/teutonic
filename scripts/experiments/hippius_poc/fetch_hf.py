"""Stage 1: snapshot_download a HF model to local disk and emit a manifest.

Usage:
    python fetch_hf.py --repo TinyLlama/TinyLlama-1.1B-Chat-v1.0 --out /workspace/hf_cache/tinyllama --run-dir runs/<id> --tier tinyllama
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

import lib


ALLOW_PATTERNS = [
    "*.safetensors",
    "config.json",
    "tokenizer*",
    "special_tokens*",
    "generation_config.json",
    "*.model",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="HF repo id")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--out", required=True, help="Local destination dir")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--tier", required=True)
    ap.add_argument("--wipe", action="store_true",
                    help="Remove --out before download (cold-cache benchmark)")
    args = ap.parse_args()

    log = lib.open_logger(args.run_dir, args.tier)

    out = Path(args.out)
    if args.wipe and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    token = os.environ.get("HF_TOKEN") or None

    log.emit("hf_dl_start", repo=args.repo, out=str(out))
    with lib.timer() as t:
        local = snapshot_download(
            repo_id=args.repo,
            revision=args.revision,
            local_dir=str(out),
            token=token,
            allow_patterns=ALLOW_PATTERNS,
            etag_timeout=int(os.environ.get("HF_HUB_ETAG_TIMEOUT", "30")),
            max_workers=int(os.environ.get("HF_HUB_DOWNLOAD_WORKERS", "16")),
        )

    files = lib.list_allowed(local)
    n_bytes = lib.total_bytes(files)
    h_safetensors = lib.sha256_dir(local)
    per_file = lib.sha256_dir_full(local)

    manifest = {
        "repo": args.repo,
        "revision": args.revision,
        "local_dir": str(local),
        "files": [
            {"rel": p.relative_to(local).as_posix(), "size": p.stat().st_size}
            for p in files
        ],
        "total_bytes": n_bytes,
        "sha256_safetensors": h_safetensors,
        "sha256_per_file": per_file,
    }
    lib.write_json(Path(args.run_dir) / f"manifest.{args.tier}.json", manifest)

    log.emit(
        "hf_dl_done",
        bytes=n_bytes,
        bytes_h=lib.humanize_bytes(n_bytes),
        seconds=round(t.elapsed, 3),
        mbps=round(lib.mbps(n_bytes, t.elapsed), 2),
        files=len(files),
        sha256=h_safetensors[:16],
    )


if __name__ == "__main__":
    main()
