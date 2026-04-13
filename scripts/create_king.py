#!/usr/bin/env python3
"""Create Teutonic-I: a randomly initialized ~1B param LLM.

Uses a modified Gemma 3 architecture (Gemma3ForCausalLM) with tweaked
dimensions so pretrained Gemma 3 weights cannot be directly submitted.
SGLang natively supports this architecture class.
"""

import argparse
import os
import shutil
import subprocess
import sys

from transformers import AutoConfig, AutoModelForCausalLM


def get_hf_token() -> str:
    result = subprocess.run(
        ["doppler", "secrets", "get", "HF_TOKEN", "--plain", "-p", "arbos", "-c", "prd"],
        capture_output=True, text=True,
    )
    token = result.stdout.strip()
    if not token or result.returncode != 0:
        sys.exit("Failed to get HF_TOKEN from Doppler")
    return token


def build_config():
    """Gemma 3-based config with tweaked dims to prevent pretrained weight reuse.

    Changes from google/gemma-3-1b-it:
      num_hidden_layers: 26 -> 24
      intermediate_size: 6912 -> 6400
    These make any pretrained Gemma 3 checkpoint incompatible (shape mismatch)
    while keeping the Gemma3ForCausalLM architecture for native SGLang support.
    """
    config = AutoConfig.for_model("gemma3_text",
        hidden_size=1152,
        num_hidden_layers=24,
        num_attention_heads=4,
        num_key_value_heads=1,
        intermediate_size=6400,
        head_dim=256,
        hidden_activation="gelu_pytorch_tanh",
        rms_norm_eps=1e-6,
        vocab_size=262144,
        max_position_embeddings=32768,
        sliding_window=512,
        sliding_window_pattern=6,
        query_pre_attn_scalar=256,
        bos_token_id=2,
        eos_token_id=106,
        pad_token_id=0,
        attention_bias=False,
        attention_dropout=0.0,
        initializer_range=0.02,
        torch_dtype="bfloat16",
    )
    config.architectures = ["Gemma3ForCausalLM"]
    return config


def main():
    parser = argparse.ArgumentParser(description="Create Teutonic-I king model")
    parser.add_argument("--repo", default="unconst/Teutonic-I", help="HF repo to upload to")
    parser.add_argument("--local-dir", default="/tmp/teutonic/king-init", help="Local save dir")
    parser.add_argument("--skip-upload", action="store_true")
    args = parser.parse_args()

    token = get_hf_token()

    print("Building custom Gemma 3 config (tweaked dims)...")
    config = build_config()

    print("Randomly initializing model...")
    model = AutoModelForCausalLM.from_config(config)
    total = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total:,} ({total/1e9:.2f}B)")

    if os.path.exists(args.local_dir):
        shutil.rmtree(args.local_dir)
    os.makedirs(args.local_dir, exist_ok=True)

    print(f"Saving model to {args.local_dir}...")
    model.save_pretrained(args.local_dir, safe_serialization=True)

    print("Downloading Gemma 3 tokenizer from unsloth mirror...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-3-1b-it", token=token)
    tokenizer.save_pretrained(args.local_dir)
    print("Saved: config.json + safetensors + tokenizer")

    if not args.skip_upload:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.create_repo(args.repo, exist_ok=True, private=False)
        api.upload_folder(
            folder_path=args.local_dir,
            repo_id=args.repo,
            commit_message="Teutonic-I: randomly initialized ~1B custom Gemma 3 architecture",
            allow_patterns=["*.safetensors", "config.json", "tokenizer*", "special_tokens*"],
        )
        print(f"Uploaded to https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
