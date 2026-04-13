"""CLI entry points for the Teutonic system."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-20s %(levelname)-7s %(message)s",
)


def _load_config():
    """Load TeutonicConfig from environment variables or a JSON file."""
    from .config import (
        TeutonicConfig, EvalConfig, BoundingBoxConfig, R2Config,
        KingConfig, ChainConfig, PodConfig,
    )

    return TeutonicConfig(
        eval=EvalConfig(
            N=int(os.getenv("TEUTONIC_EVAL_N", "10000")),
            alpha=float(os.getenv("TEUTONIC_EVAL_ALPHA", "0.01")),
            sequence_length=int(os.getenv("TEUTONIC_SEQUENCE_LENGTH", "2048")),
        ),
        bounding_box=BoundingBoxConfig(
            max_linf=float(os.getenv("TEUTONIC_BBOX_MAX_LINF", "0.5")),
            max_l2_global=float(os.getenv("TEUTONIC_BBOX_MAX_L2_GLOBAL", "0")) or None,
        ),
        r2=R2Config(
            endpoint_url=os.getenv("TEUTONIC_R2_ENDPOINT", ""),
            bucket_name=os.getenv("TEUTONIC_R2_BUCKET", ""),
            access_key_id=os.getenv("TEUTONIC_R2_ACCESS_KEY", ""),
            secret_access_key=os.getenv("TEUTONIC_R2_SECRET_KEY", ""),
        ),
        king=KingConfig(
            hf_repo=os.getenv("TEUTONIC_KING_REPO", ""),
            hf_token=os.getenv("HF_TOKEN", ""),
            local_cache_dir=os.getenv("TEUTONIC_CACHE_DIR", "/tmp/teutonic/king"),
        ),
        chain=ChainConfig(
            netuid=int(os.getenv("TEUTONIC_NETUID", "3")),
            network=os.getenv("TEUTONIC_NETWORK", "finney"),
            wallet_name=os.getenv("TEUTONIC_WALLET_NAME", "default"),
            wallet_hotkey=os.getenv("TEUTONIC_WALLET_HOTKEY", "default"),
        ),
        pod=PodConfig(
            gpu_type=os.getenv("TEUTONIC_GPU_TYPE", "rtx4090"),
        ),
        poll_interval_s=int(os.getenv("TEUTONIC_POLL_INTERVAL", "12")),
    )


def run_validator():
    """Run the Teutonic validator coordinator."""
    from .validator import Validator

    config = _load_config()
    validator = Validator(config)
    validator.run()


def run_miner():
    """Run the Teutonic reference miner."""
    parser = argparse.ArgumentParser(description="Teutonic Reference Miner")
    parser.add_argument("--king-repo", required=True, help="HF repo of the current king")
    parser.add_argument("--miner-repo", required=True, help="Your HF repo to upload the challenger")
    parser.add_argument("--dataset", default="", help="Path to tokenized .npy dataset shard")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--steps", type=int, default=100, help="Training steps")
    parser.add_argument("--seq-len", type=int, default=2048, help="Sequence length")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--netuid", type=int, default=3)
    parser.add_argument("--network", default="finney")
    parser.add_argument("--wallet-name", default="default")
    parser.add_argument("--wallet-hotkey", default="default")
    args = parser.parse_args()

    from .miner import Miner

    miner = Miner(
        king_repo=args.king_repo,
        miner_repo=args.miner_repo,
        netuid=args.netuid,
        network=args.network,
        wallet_name=args.wallet_name,
        wallet_hotkey=args.wallet_hotkey,
        hf_token=os.getenv("HF_TOKEN", ""),
        learning_rate=args.lr,
        train_steps=args.steps,
        sequence_length=args.seq_len,
        batch_size=args.batch_size,
        dataset_path=args.dataset,
    )
    miner.run()


def seed_king():
    """Upload an initial seed model to the king repo."""
    parser = argparse.ArgumentParser(description="Seed the king model")
    parser.add_argument("--model-dir", required=True, help="Path to the seed model directory")
    parser.add_argument("--king-repo", required=True, help="HF repo for the king")
    args = parser.parse_args()

    from .king import KingManager

    mgr = KingManager(
        hf_repo=args.king_repo,
        cache_dir="/tmp/teutonic/king",
        hf_token=os.getenv("HF_TOKEN", ""),
    )
    king_hash = mgr.upload_seed(args.model_dir)
    print(f"Seed king uploaded to {args.king_repo}")
    print(f"King hash: {king_hash}")


def show_state():
    """Show the current validator state from R2."""
    config = _load_config()

    from .r2 import R2Client

    r2 = R2Client(config.r2)

    state = r2.get_json("state/validator_state.json")
    if state:
        print(json.dumps(state, indent=2))
    else:
        print("No validator state found in R2.")

    king = r2.get_json("king/current.json")
    if king:
        print("\nCurrent King:")
        print(json.dumps(king, indent=2))


def run_dataset():
    """Dataset management commands for miners."""
    parser = argparse.ArgumentParser(description="Teutonic Dataset Tools")
    sub = parser.add_subparsers(dest="action")

    dl = sub.add_parser("download", help="Download dataset shards from R2 for training")
    dl.add_argument("--shards", type=int, default=1, help="Number of shards to download")
    dl.add_argument("--output", default="./data", help="Output directory")
    dl.add_argument("--seed", type=int, default=None, help="RNG seed for shard selection (random if omitted)")
    dl.add_argument("--indices", type=str, default="", help="Comma-separated shard indices to download (overrides --shards)")

    info = sub.add_parser("info", help="Show dataset manifest info")

    args = parser.parse_args()

    config = _load_config()

    from .r2 import R2Client
    from .dataset import load_manifest_cached, download_shard_from_r2

    r2 = R2Client(config.r2)

    if args.action == "info":
        manifest = load_manifest_cached(r2)
        print(f"Dataset v{manifest.version}")
        print(f"  Tokenizer: {manifest.tokenizer}")
        print(f"  Dtype: {manifest.dtype}")
        print(f"  Total tokens: {manifest.total_tokens:,}")
        print(f"  Total shards: {manifest.total_shards}")
        total_bytes = sum(s.get("size_bytes", 0) for s in manifest.shards)
        print(f"  Total size: {total_bytes / 1e9:.1f} GB")
        print(f"\nShards:")
        for i, s in enumerate(manifest.shards):
            size_gb = s.get("size_bytes", 0) / 1e9
            print(f"  [{i:4d}] {s['key']}  {s['n_tokens']:>12,} tokens  {size_gb:.2f} GB")
        return

    if args.action == "download":
        import numpy as np
        from pathlib import Path

        manifest = load_manifest_cached(r2)
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        if args.indices:
            selected = [int(x.strip()) for x in args.indices.split(",")]
            for idx in selected:
                if idx < 0 or idx >= manifest.total_shards:
                    print(f"Error: shard index {idx} out of range [0, {manifest.total_shards})")
                    sys.exit(1)
        else:
            n = min(args.shards, manifest.total_shards)
            seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(8), "little")
            rng = np.random.Generator(np.random.PCG64(seed))
            selected = rng.choice(manifest.total_shards, size=n, replace=False).tolist()
            print(f"Selected {n} shard(s) (seed={seed}): {selected}")

        for idx in selected:
            shard = manifest.shards[idx]
            shard_key = shard["key"]
            filename = shard_key.split("/")[-1]
            local_path = output_dir / filename

            print(f"Downloading shard {idx}: {shard_key} -> {local_path}")
            download_shard_from_r2(r2, shard_key, local_path)
            print(f"  Done: {local_path.stat().st_size / 1e9:.2f} GB")

        print(f"\n{len(selected)} shard(s) downloaded to {output_dir}/")
        return

    parser.print_help()


def main():
    parser = argparse.ArgumentParser(description="Teutonic")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("validator", help="Run the validator coordinator")
    sub.add_parser("miner", help="Run the reference miner")
    sub.add_parser("seed", help="Upload initial seed king model")
    sub.add_parser("state", help="Show current validator state from R2")
    sub.add_parser("dataset", help="Dataset management (download, info)")

    args, remaining = parser.parse_known_args()

    if args.command == "validator":
        sys.argv = [sys.argv[0]] + remaining
        run_validator()
    elif args.command == "miner":
        sys.argv = [sys.argv[0]] + remaining
        run_miner()
    elif args.command == "seed":
        sys.argv = [sys.argv[0]] + remaining
        seed_king()
    elif args.command == "state":
        sys.argv = [sys.argv[0]] + remaining
        show_state()
    elif args.command == "dataset":
        sys.argv = [sys.argv[0]] + remaining
        run_dataset()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
