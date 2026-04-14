#!/usr/bin/env python3
"""Extended gaming experiment: quantify how easy it is to game the evaluation.

Extends experiment_copy_resistance.py with:
1. Monte Carlo win-rate estimation (many trials per noise scale)
2. Targeted perturbation strategies (layer-selective, magnitude-scaled, bias-only)
3. Shard sensitivity analysis
4. Economic breakeven analysis
"""
import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval_torch import R2, run_bootstrap_test, parse_gpu_ids, load_model, compute_batch_losses

log = logging.getLogger("gaming")


class LocalMultiGPUEvaluator:
    def __init__(self, model_dir, gpu_ids, label):
        from concurrent.futures import ThreadPoolExecutor
        self.gpu_ids = gpu_ids
        self.models = {}
        self.devices = {}
        for gid in gpu_ids:
            self.models[gid] = load_model(model_dir, f"cuda:{gid}", f"{label}-gpu{gid}")
            self.devices[gid] = f"cuda:{gid}"
        self.pool = ThreadPoolExecutor(max_workers=len(gpu_ids))
        log.info("%s evaluator ready: %d GPUs %s", label, len(gpu_ids), gpu_ids)

    def compute_losses(self, token_batches):
        from concurrent.futures import as_completed
        n_gpus = len(self.gpu_ids)
        if not token_batches:
            return []
        per_gpu = [[] for _ in range(n_gpus)]
        idx_map = [[] for _ in range(n_gpus)]
        for i, batch in enumerate(token_batches):
            g = i % n_gpus
            per_gpu[g].append(batch)
            idx_map[g].append(i)
        futures = {}
        for g_idx, gid in enumerate(self.gpu_ids):
            if per_gpu[g_idx]:
                fut = self.pool.submit(compute_batch_losses, self.models[gid], per_gpu[g_idx], self.devices[gid])
                futures[fut] = g_idx
        results = [None] * len(token_batches)
        for fut in as_completed(futures):
            g_idx = futures[fut]
            losses = fut.result()
            for local_i, global_i in enumerate(idx_map[g_idx]):
                results[global_i] = losses[local_i]
        return results

    def shutdown(self):
        self.pool.shutdown(wait=False)


def perturb_model_uniform(src_dir, dst_dir, noise_scale, seed=None):
    """Standard uniform gaussian noise on all float params."""
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)
    rng = np.random.default_rng(seed)
    for st_file in sorted(Path(dst_dir).glob("*.safetensors")):
        sd = load_file(str(st_file))
        new_sd = {}
        for name, tensor in sd.items():
            if tensor.dtype in (torch.bfloat16, torch.float16, torch.float32):
                gen = torch.Generator()
                gen.manual_seed(int(rng.integers(0, 2**63)))
                noise = torch.randn(tensor.shape, generator=gen, dtype=torch.float32) * noise_scale
                new_sd[name] = (tensor.float() + noise).to(tensor.dtype)
            else:
                new_sd[name] = tensor
        save_file(new_sd, str(st_file))
    return dst_dir


def perturb_model_layer_selective(src_dir, dst_dir, noise_scale, layer_filter, seed=None):
    """Only perturb parameters matching layer_filter pattern."""
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)
    rng = np.random.default_rng(seed)
    perturbed_count = 0
    total_count = 0
    for st_file in sorted(Path(dst_dir).glob("*.safetensors")):
        sd = load_file(str(st_file))
        new_sd = {}
        for name, tensor in sd.items():
            total_count += 1
            if tensor.dtype in (torch.bfloat16, torch.float16, torch.float32) and layer_filter(name):
                gen = torch.Generator()
                gen.manual_seed(int(rng.integers(0, 2**63)))
                noise = torch.randn(tensor.shape, generator=gen, dtype=torch.float32) * noise_scale
                new_sd[name] = (tensor.float() + noise).to(tensor.dtype)
                perturbed_count += 1
            else:
                new_sd[name] = tensor
        save_file(new_sd, str(st_file))
    log.info("layer-selective: perturbed %d/%d params", perturbed_count, total_count)
    return dst_dir


def perturb_model_magnitude_scaled(src_dir, dst_dir, noise_scale, seed=None):
    """Scale noise proportional to parameter magnitude: param + param * noise_scale * randn."""
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)
    rng = np.random.default_rng(seed)
    for st_file in sorted(Path(dst_dir).glob("*.safetensors")):
        sd = load_file(str(st_file))
        new_sd = {}
        for name, tensor in sd.items():
            if tensor.dtype in (torch.bfloat16, torch.float16, torch.float32):
                gen = torch.Generator()
                gen.manual_seed(int(rng.integers(0, 2**63)))
                noise = torch.randn(tensor.shape, generator=gen, dtype=torch.float32)
                new_sd[name] = (tensor.float() * (1.0 + noise_scale * noise)).to(tensor.dtype)
            else:
                new_sd[name] = tensor
        save_file(new_sd, str(st_file))
    return dst_dir


def perturb_model_bias_only(src_dir, dst_dir, noise_scale, seed=None):
    """Only perturb bias parameters."""
    return perturb_model_layer_selective(
        src_dir, dst_dir, noise_scale,
        layer_filter=lambda name: "bias" in name.lower(),
        seed=seed,
    )


def perturb_model_embed_only(src_dir, dst_dir, noise_scale, seed=None):
    """Only perturb embedding and lm_head layers."""
    return perturb_model_layer_selective(
        src_dir, dst_dir, noise_scale,
        layer_filter=lambda name: any(k in name.lower() for k in ["embed", "lm_head"]),
        seed=seed,
    )


def perturb_model_early_layers(src_dir, dst_dir, noise_scale, seed=None):
    """Only perturb first quarter of transformer layers."""
    return perturb_model_layer_selective(
        src_dir, dst_dir, noise_scale,
        layer_filter=lambda name: any(f"layers.{i}." in name for i in range(6)),
        seed=seed,
    )


def perturb_model_late_layers(src_dir, dst_dir, noise_scale, seed=None):
    """Only perturb last quarter of transformer layers."""
    return perturb_model_layer_selective(
        src_dir, dst_dir, noise_scale,
        layer_filter=lambda name: any(f"layers.{i}." in name for i in range(18, 24)),
        seed=seed,
    )


STRATEGIES = {
    "uniform": perturb_model_uniform,
    "magnitude_scaled": perturb_model_magnitude_scaled,
    "bias_only": perturb_model_bias_only,
    "embed_only": perturb_model_embed_only,
    "early_layers": perturb_model_early_layers,
    "late_layers": perturb_model_late_layers,
}


def run_single_trial(king_eval, r2, manifest, king_dir, strategy_name, strategy_fn,
                     noise_scale, seed, gpu_ids, args, trial_idx):
    """Run a single perturbation trial and return the verdict dict."""
    challenger_dir = f"/tmp/experiment/challenger_{strategy_name}_n{noise_scale}_s{seed}"

    strategy_fn(king_dir, challenger_dir, noise_scale, seed=seed)

    mid = len(gpu_ids) // 2
    challenger_gpus = gpu_ids[mid:]
    chall_eval = LocalMultiGPUEvaluator(challenger_dir, challenger_gpus,
                                        f"chall-{strategy_name}-{noise_scale}-{seed}")

    n_shards = manifest["total_shards"]
    shard_idx = abs(hash(f"{strategy_name}:{noise_scale}:{seed}:{trial_idx}")) % n_shards
    shard_key = manifest["shards"][shard_idx]["key"]
    seed_str = f"{strategy_name}:{noise_scale}:{seed}:{trial_idx}"

    verdict = run_bootstrap_test(
        king_eval, chall_eval, r2, shard_key,
        args.n, args.alpha, args.delta, args.seq_len, args.batch_size,
        seed_str, n_bootstrap=args.n_bootstrap,
    )

    verdict["strategy"] = strategy_name
    verdict["noise_scale"] = noise_scale
    verdict["seed"] = seed
    verdict["trial_idx"] = trial_idx
    verdict["shard_idx"] = shard_idx

    chall_eval.shutdown()
    del chall_eval
    torch.cuda.empty_cache()
    shutil.rmtree(challenger_dir, ignore_errors=True)

    return verdict


def run_experiment(args):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    gpu_ids = parse_gpu_ids(args.gpus)
    log.info("GPUs: %s", gpu_ids)

    r2 = R2()
    manifest = r2.get("dataset/v1/manifest.json")
    if not manifest:
        log.error("could not fetch dataset manifest")
        sys.exit(1)
    n_shards = manifest["total_shards"]
    log.info("dataset: %d shards", n_shards)

    king_dir = "/tmp/experiment/king"
    if not os.path.exists(king_dir) or args.redownload:
        from huggingface_hub import snapshot_download
        log.info("downloading king model from %s", args.king_repo)
        if os.path.exists(king_dir):
            shutil.rmtree(king_dir)
        snapshot_download(args.king_repo, local_dir=king_dir,
                         token=os.environ.get("HF_TOKEN") or None)
    else:
        log.info("using cached king at %s", king_dir)

    mid = len(gpu_ids) // 2
    king_gpus = gpu_ids[:mid]
    log.info("king GPUs: %s, challenger GPUs: %s", king_gpus, gpu_ids[mid:])

    results = []
    strategies_to_test = [s.strip() for s in args.strategies.split(",")]
    noise_scales = [float(x) for x in args.noise_scales.split(",")]

    for strategy_name in strategies_to_test:
        if strategy_name not in STRATEGIES:
            log.warning("unknown strategy: %s, skipping", strategy_name)
            continue

        strategy_fn = STRATEGIES[strategy_name]
        log.info("=" * 60)
        log.info("STRATEGY: %s", strategy_name)
        log.info("=" * 60)

        for noise_scale in noise_scales:
            king_eval = LocalMultiGPUEvaluator(king_dir, king_gpus, "king")

            for seed in range(args.trials_per_config):
                log.info("--- %s | noise=%.6f | trial=%d/%d ---",
                         strategy_name, noise_scale, seed + 1, args.trials_per_config)

                verdict = run_single_trial(
                    king_eval, r2, manifest, king_dir, strategy_name, strategy_fn,
                    noise_scale, seed, gpu_ids, args, trial_idx=seed,
                )
                results.append(verdict)

                log.info("RESULT %s noise=%.6f trial=%d: verdict=%s mu_hat=%.6f lcb=%.6f "
                         "king_loss=%.6f chall_loss=%.6f",
                         strategy_name, noise_scale, seed,
                         verdict["verdict"], verdict["mu_hat"], verdict["lcb"],
                         verdict["avg_king_loss"], verdict["avg_challenger_loss"])

            king_eval.shutdown()
            del king_eval
            torch.cuda.empty_cache()

    # --- Shard sensitivity (using uniform noise at one scale) ---
    if not args.skip_shard_sensitivity:
        log.info("=" * 60)
        log.info("SHARD SENSITIVITY ANALYSIS")
        log.info("=" * 60)

        shard_noise = float(args.shard_sensitivity_noise)
        challenger_dir = "/tmp/experiment/challenger_shard_test"
        perturb_model_uniform("/tmp/experiment/king", challenger_dir, shard_noise, seed=42)

        king_eval = LocalMultiGPUEvaluator(king_dir, king_gpus, "king")
        chall_eval = LocalMultiGPUEvaluator(challenger_dir, gpu_ids[mid:], "chall-shard")

        shard_indices = np.linspace(0, n_shards - 1, min(20, n_shards), dtype=int).tolist()
        for si, shard_idx in enumerate(shard_indices):
            shard_key = manifest["shards"][shard_idx]["key"]
            seed_str = f"shard_sensitivity:{shard_idx}"

            verdict = run_bootstrap_test(
                king_eval, chall_eval, r2, shard_key,
                args.n, args.alpha, args.delta, args.seq_len, args.batch_size,
                seed_str, n_bootstrap=args.n_bootstrap,
            )
            verdict["experiment"] = "shard_sensitivity"
            verdict["shard_idx"] = shard_idx
            verdict["noise_scale"] = shard_noise
            verdict["strategy"] = "uniform"
            results.append(verdict)

            log.info("SHARD %d/%d (idx=%d): verdict=%s mu_hat=%.6f lcb=%.6f",
                     si + 1, len(shard_indices), shard_idx,
                     verdict["verdict"], verdict["mu_hat"], verdict["lcb"])

        chall_eval.shutdown()
        king_eval.shutdown()
        del chall_eval, king_eval
        torch.cuda.empty_cache()
        shutil.rmtree(challenger_dir, ignore_errors=True)

    # --- Save results ---
    out_path = args.output
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("Results saved to %s (%d entries)", out_path, len(results))

    # --- Analysis ---
    print_analysis(results, noise_scales, strategies_to_test)


def print_analysis(results, noise_scales, strategies):
    """Print economic analysis of results."""
    print()
    print("=" * 80)
    print("GAMING ANALYSIS REPORT")
    print("=" * 80)

    strategy_results = [r for r in results if r.get("experiment") != "shard_sensitivity"]
    shard_results = [r for r in results if r.get("experiment") == "shard_sensitivity"]

    print("\n--- Win Rate by Strategy & Noise Scale ---")
    print(f"{'Strategy':<20} {'Noise':>10} {'Wins':>6} {'Total':>6} {'Win%':>8} "
          f"{'E[cost]':>10} {'mu_hat_avg':>12} {'lcb_avg':>12}")
    print("-" * 100)

    for strat in strategies:
        for ns in noise_scales:
            subset = [r for r in strategy_results
                      if r.get("strategy") == strat and r.get("noise_scale") == ns]
            if not subset:
                continue
            wins = sum(1 for r in subset if r["accepted"])
            total = len(subset)
            win_rate = wins / total if total > 0 else 0
            expected_cost = 1.0 / win_rate if win_rate > 0 else float("inf")
            avg_mu = np.mean([r["mu_hat"] for r in subset])
            avg_lcb = np.mean([r["lcb"] for r in subset])

            print(f"{strat:<20} {ns:>10.4f} {wins:>6} {total:>6} {win_rate*100:>7.1f}% "
                  f"{'$'+f'{expected_cost:.0f}':>10} {avg_mu:>12.6f} {avg_lcb:>12.6f}")

    if shard_results:
        print("\n--- Shard Sensitivity ---")
        mu_hats = [r["mu_hat"] for r in shard_results]
        lcbs = [r["lcb"] for r in shard_results]
        wins = sum(1 for r in shard_results if r["accepted"])
        print(f"Shards tested: {len(shard_results)}")
        print(f"Wins: {wins}/{len(shard_results)} ({100*wins/len(shard_results):.1f}%)")
        print(f"mu_hat range: [{min(mu_hats):.6f}, {max(mu_hats):.6f}]")
        print(f"mu_hat std:   {np.std(mu_hats):.6f}")
        print(f"lcb range:    [{min(lcbs):.6f}, {max(lcbs):.6f}]")

    print("\n--- Economic Summary ---")
    all_wins = sum(1 for r in strategy_results if r["accepted"])
    all_total = len(strategy_results)
    if all_total > 0:
        overall_rate = all_wins / all_total
        print(f"Overall win rate: {all_wins}/{all_total} ({100*overall_rate:.1f}%)")
        if overall_rate > 0:
            print(f"Expected submissions to win (any strategy): ${1/overall_rate:.0f}")
        else:
            print("No wins observed -- system appears robust to these perturbations")

    best_strat = None
    best_rate = 0
    for strat in strategies:
        for ns in noise_scales:
            subset = [r for r in strategy_results
                      if r.get("strategy") == strat and r.get("noise_scale") == ns]
            if not subset:
                continue
            rate = sum(1 for r in subset if r["accepted"]) / len(subset)
            if rate > best_rate:
                best_rate = rate
                best_strat = f"{strat} @ noise={ns}"

    if best_strat:
        print(f"Best strategy: {best_strat} (win rate: {best_rate*100:.1f}%, "
              f"expected cost: ${1/best_rate:.0f})")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Extended gaming experiment")
    parser.add_argument("--king-repo", default="kt3202/Teutonic-I-test")
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--gpus", default="auto")
    parser.add_argument("--strategies", default="uniform,magnitude_scaled,bias_only,embed_only,early_layers,late_layers")
    parser.add_argument("--noise-scales", default="0.0001,0.001,0.005,0.01")
    parser.add_argument("--trials-per-config", type=int, default=10,
                        help="Number of random seeds per (strategy, noise_scale)")
    parser.add_argument("--skip-shard-sensitivity", action="store_true")
    parser.add_argument("--shard-sensitivity-noise", default="0.001")
    parser.add_argument("--redownload", action="store_true")
    parser.add_argument("--output", default="/tmp/experiment/results_gaming.json")
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
