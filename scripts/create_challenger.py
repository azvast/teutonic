#!/usr/bin/env python3
"""Create a challenger model by adding small noise to the king (Teutonic-I).

The perturbation stays well within the bounding box (max_linf=0.5, using epsilon=0.01).
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def get_hf_token() -> str:
    result = subprocess.run(
        ["doppler", "secrets", "get", "HF_TOKEN", "--plain", "-p", "arbos", "-c", "prd"],
        capture_output=True, text=True,
    )
    token = result.stdout.strip()
    if not token or result.returncode != 0:
        sys.exit("Failed to get HF_TOKEN from Doppler")
    return token


def main():
    parser = argparse.ArgumentParser(description="Create perturbed challenger from king")
    parser.add_argument("--king-repo", default="unconst/Teutonic-I")
    parser.add_argument("--challenger-repo", default="unconst/Teutonic-I-challenger")
    parser.add_argument("--king-dir", default="/tmp/teutonic/king-init")
    parser.add_argument("--challenger-dir", default="/tmp/teutonic/challenger-init")
    parser.add_argument("--epsilon", type=float, default=0.01, help="Max perturbation per weight")
    parser.add_argument("--skip-upload", action="store_true")
    args = parser.parse_args()

    king_dir = Path(args.king_dir)
    challenger_dir = Path(args.challenger_dir)

    if not king_dir.exists() or not list(king_dir.glob("*.safetensors")):
        print(f"King not found at {king_dir}, downloading from HF...")
        token = get_hf_token()
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=args.king_repo,
            local_dir=str(king_dir),
            token=token,
            allow_patterns=["*.safetensors", "config.json"],
        )

    challenger_dir.mkdir(parents=True, exist_ok=True)

    # Copy config.json
    for f in king_dir.glob("config.json"):
        shutil.copy2(f, challenger_dir / f.name)

    # Load, perturb, save safetensors
    for sf in sorted(king_dir.glob("*.safetensors")):
        print(f"Perturbing {sf.name} (epsilon={args.epsilon})...")
        sd = load_file(str(sf))
        perturbed = {}
        for name, tensor in sd.items():
            noise = torch.empty_like(tensor).uniform_(-args.epsilon, args.epsilon)
            perturbed[name] = tensor + noise
        save_file(perturbed, str(challenger_dir / sf.name))

    print(f"Challenger saved to {challenger_dir}")

    # Verify delta is within bounds
    king_sd = load_file(str(list(king_dir.glob("*.safetensors"))[0]))
    chal_sd = load_file(str(list(challenger_dir.glob("*.safetensors"))[0]))
    sample_key = list(king_sd.keys())[0]
    delta = (chal_sd[sample_key].float() - king_sd[sample_key].float()).abs()
    print(f"Sample delta check ({sample_key}): max_linf={delta.max().item():.6f}, mean={delta.mean().item():.6f}")

    if not args.skip_upload:
        token = get_hf_token()
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.create_repo(args.challenger_repo, exist_ok=True, private=False)
        api.upload_folder(
            folder_path=str(challenger_dir),
            repo_id=args.challenger_repo,
            commit_message=f"Teutonic-I challenger: epsilon={args.epsilon} perturbation",
            allow_patterns=["*.safetensors", "config.json"],
        )
        print(f"Uploaded to https://huggingface.co/{args.challenger_repo}")


if __name__ == "__main__":
    main()
