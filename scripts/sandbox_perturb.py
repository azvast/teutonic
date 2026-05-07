#!/usr/bin/env python3
"""Sandbox helper for the Teutonic-LXXX soak.

Pulls a base king from HF, perturbs every floating-point safetensor with
Gaussian noise (mirrors miner.py:187-204 — same dtype handling), and pushes
the perturbed copy to a target repo. Intended ONLY for the sandbox smoke:
generates a deterministic-but-non-trivial mock challenger so the eval-server
sharded path has two distinct repos to load.

NOT a chain-mining script: no on-chain reveal, no coldkey gating, no
config.json validation. For real mining use scripts/mining/train_challenger.py
or miner.py.

Run on the sandbox box (~160 GB download + ~160 GB upload, plan ~1-2 hr):
    source /workspace/teutonic/.venv/bin/activate
    HF_HOME=/workspace/hf-cache \
    python scripts/sandbox_perturb.py \
        --base unconst/Teutonic-LXXX-mock-king \
        --upload-repo unconst/Teutonic-LXXX-mock-chall \
        --noise 1e-4
"""
from __future__ import annotations

import argparse
import logging
import os

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

import shutil
import time
from pathlib import Path

import numpy as np  # noqa: F401  (parity with miner.py imports for any future seed plumbing)
import torch
from huggingface_hub import HfApi, snapshot_download
from safetensors.torch import load_file, save_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [sandbox_perturb] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sandbox_perturb")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True,
                    help="HF repo of the base king to perturb")
    ap.add_argument("--base-revision", default=None,
                    help="pinned commit SHA of the base king (default: HEAD)")
    ap.add_argument("--upload-repo", required=True,
                    help="HF repo to push the perturbed challenger to")
    ap.add_argument("--noise", type=float, default=1e-4,
                    help="Gaussian noise stdev (matches miner.py default scale)")
    ap.add_argument("--seed", type=int, default=int(time.time()))
    ap.add_argument("--workdir", default="/workspace/sandbox-mock",
                    help="local scratch dir")
    ap.add_argument("--private", action="store_true",
                    help="create the upload repo private (default: public)")
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    args = ap.parse_args()

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    base_dir = workdir / "base"
    chall_dir = workdir / "challenger"

    if base_dir.exists():
        log.info("clearing %s", base_dir)
        shutil.rmtree(base_dir)
    if chall_dir.exists():
        log.info("clearing %s", chall_dir)
        shutil.rmtree(chall_dir)

    log.info("downloading base %s@%s -> %s",
             args.base, (args.base_revision or "HEAD")[:12], base_dir)
    t0 = time.time()
    snapshot_download(args.base, local_dir=str(base_dir),
                      revision=args.base_revision,
                      token=args.hf_token or None,
                      max_workers=16)
    log.info("download done in %.1fs", time.time() - t0)

    log.info("copying base -> challenger working tree (~160 GiB on disk)")
    t1 = time.time()
    shutil.copytree(base_dir, chall_dir)
    log.info("copy done in %.1fs", time.time() - t1)

    st_files = sorted(chall_dir.glob("*.safetensors"))
    log.info("perturbing %d safetensors files with noise stdev=%.3g (seed=%d)",
             len(st_files), args.noise, args.seed)
    torch.manual_seed(args.seed)

    for st_file in st_files:
        t_file = time.time()
        sd = load_file(str(st_file))
        new_sd = {}
        n_tensors = 0
        n_floats = 0
        for name, tensor in sd.items():
            n_tensors += 1
            if tensor.dtype in (torch.bfloat16, torch.float16, torch.float32):
                # bf16-direct (no fp32 cast): saves ~3x memory + ~2x wall vs
                # the original fp32-cast path, which mattered when we were
                # perturbing 4 shards × 50 GiB ≈ 200 GiB of bf16 weights.
                noise = torch.randn(tensor.shape, dtype=tensor.dtype) * args.noise
                new_sd[name] = tensor + noise
                n_floats += 1
            else:
                new_sd[name] = tensor
        save_file(new_sd, str(st_file))
        log.info("  %s: %d tensors, %d perturbed (%.1fs)",
                 st_file.name, n_tensors, n_floats, time.time() - t_file)

    api = HfApi(token=args.hf_token or None)
    log.info("creating/updating repo %s (private=%s)",
             args.upload_repo, args.private)
    api.create_repo(args.upload_repo, exist_ok=True,
                    private=args.private, repo_type="model")
    log.info("uploading %s -> %s", chall_dir, args.upload_repo)
    t2 = time.time()
    api.upload_folder(
        folder_path=str(chall_dir),
        repo_id=args.upload_repo,
        commit_message=(
            f"Sandbox mock challenger: {args.base} + N(0, {args.noise}^2) seed={args.seed}"
        ),
        allow_patterns=[
            "*.safetensors", "*.json", "tokenizer*", "special_tokens*",
            "vocab*", "merges*",
        ],
    )
    log.info("upload done in %.1fs", time.time() - t2)
    info = api.repo_info(args.upload_repo)
    log.info("uploaded -> https://huggingface.co/%s @ %s",
             args.upload_repo, info.sha[:12])


if __name__ == "__main__":
    main()
