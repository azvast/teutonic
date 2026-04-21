#!/usr/bin/env python3
"""Patch the live king state to replace king_hash="seed" with the real
sha256 of the seed king's safetensors.

Computes hash via sha256_dir() (same as miner.sha256_dir), then rewrites:
  - state/validator_state.json    (R2 bucket: constantinople)
  - king/current.json             (R2 bucket: constantinople)
  - dashboard.json                (Hippius bucket: teutonic-sn3)

Run this with the same env as the validator (pm2 doppler-injected vars):
  doppler run -- python3 teutonic/scripts/patch_seed_king_hash.py
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import boto3
from botocore.client import Config as BotoConfig
from huggingface_hub import snapshot_download

R2_ENDPOINT  = os.environ["TEUTONIC_R2_ENDPOINT"]
R2_BUCKET    = os.environ["TEUTONIC_R2_BUCKET"]
R2_ACCESS    = os.environ["TEUTONIC_R2_ACCESS_KEY"]
R2_SECRET    = os.environ["TEUTONIC_R2_SECRET_KEY"]
HIP_ENDPOINT = os.environ.get("TEUTONIC_HIPPIUS_ENDPOINT", "https://s3.hippius.com")
HIP_BUCKET   = os.environ.get("TEUTONIC_HIPPIUS_BUCKET", "teutonic-sn3")
HIP_ACCESS   = os.environ.get("TEUTONIC_HIPPIUS_ACCESS_KEY", "")
HIP_SECRET   = os.environ.get("TEUTONIC_HIPPIUS_SECRET_KEY", "")
HF_TOKEN     = os.environ.get("HF_TOKEN") or None


def sha256_dir(path: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(path.glob("*.safetensors")):
        with open(p, "rb") as f:
            while chunk := f.read(1 << 20):
                h.update(chunk)
    return h.hexdigest()


def main():
    r2 = boto3.client(
        "s3", endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_ACCESS, aws_secret_access_key=R2_SECRET,
        region_name="auto",
        config=BotoConfig(
            signature_version="s3v4",
            retries={"max_attempts": 3, "mode": "adaptive"},
            s3={"addressing_style": "path"},
        ),
    )
    hip = None
    if HIP_ACCESS and HIP_SECRET:
        hip = boto3.client(
            "s3", endpoint_url=HIP_ENDPOINT,
            aws_access_key_id=HIP_ACCESS, aws_secret_access_key=HIP_SECRET,
            region_name="decentralized",
            config=BotoConfig(
                signature_version="s3v4",
                retries={"max_attempts": 3, "mode": "adaptive"},
                s3={"addressing_style": "path"},
            ),
        )

    state = json.loads(r2.get_object(Bucket=R2_BUCKET, Key="state/validator_state.json")["Body"].read())
    king  = state["king"]
    repo  = king["hf_repo"]
    rev   = king["king_revision"]
    print(f"[+] live king: {repo}@{rev[:12]}  king_hash={king['king_hash']!r}  reign={king['reign_number']}")

    if king["king_hash"] != "seed" and len(king["king_hash"]) == 64:
        print(f"[!] king_hash already a real sha256, nothing to do: {king['king_hash']}")
        return 0

    dst = Path("/tmp/teutonic_seed_king_hash")
    dst.mkdir(parents=True, exist_ok=True)
    print(f"[+] downloading {repo}@{rev[:12]} safetensors to {dst} ...")
    snapshot_download(repo, local_dir=str(dst), token=HF_TOKEN, revision=rev,
                      allow_patterns=["*.safetensors"])
    real_hash = sha256_dir(dst)
    print(f"[+] computed king_hash = {real_hash}")
    print(f"[+] first 16 = {real_hash[:16]}")

    king["king_hash"] = real_hash
    state["king"] = king
    state["updated_at"] = state.get("updated_at")

    body = json.dumps(state, default=str).encode()
    r2.put_object(Bucket=R2_BUCKET, Key="state/validator_state.json",
                  Body=body, ContentType="application/json")
    print(f"[+] wrote state/validator_state.json ({len(body)} B) to {R2_BUCKET}")

    body = json.dumps(king, default=str).encode()
    r2.put_object(Bucket=R2_BUCKET, Key="king/current.json",
                  Body=body, ContentType="application/json")
    print(f"[+] wrote king/current.json ({len(body)} B) to {R2_BUCKET}")

    if hip:
        try:
            dash = json.loads(hip.get_object(Bucket=HIP_BUCKET, Key="dashboard.json")["Body"].read())
            dash["king"]["king_hash"] = real_hash
            body = json.dumps(dash, default=str).encode()
            hip.put_object(Bucket=HIP_BUCKET, Key="dashboard.json",
                           Body=body, ContentType="application/json")
            print(f"[+] patched dashboard.json on Hippius ({len(body)} B)")
        except Exception as exc:
            print(f"[!] could not patch dashboard.json: {exc}")
    else:
        print("[!] no Hippius creds in env; dashboard.json not patched (validator will refresh it)")

    print("[ok] done. Validator must be restarted (pm2 reload teutonic-validator)")
    print("    to load the corrected king_hash into memory; otherwise it will keep")
    print("    using the in-process 'seed' value and re-flush it on next set_king.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
