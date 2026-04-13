#!/usr/bin/env python3
"""Optimized mock duel: batched sign test between king and challenger via vLLM.

Key optimizations over the original mock_duel.py:
  1. Async HTTP requests with configurable concurrency
  2. Pre-computes all losses for one model before switching to the other
  3. Supports both single-server (swap models) and dual-server modes
  4. Detailed timing and throughput statistics
"""

import argparse
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field

import httpx
import numpy as np
from scipy.stats import binom

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
logger = logging.getLogger("mock_duel_fast")


async def compute_loss_async(
    client: httpx.AsyncClient, base_url: str, model: str, token_ids: list[int],
) -> tuple[float, float]:
    """Returns (loss, request_time)."""
    t0 = time.time()
    resp = await client.post(
        f"{base_url}/v1/completions",
        json={
            "model": model,
            "prompt": token_ids,
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": 1,
            "echo": True,
        },
        timeout=120.0,
    )
    resp.raise_for_status()
    elapsed = time.time() - t0
    data = resp.json()
    logprobs_list = data["choices"][0]["logprobs"]["token_logprobs"]
    valid = [lp for lp in logprobs_list[1:-1] if lp is not None]
    if not valid:
        return float("nan"), elapsed
    return -sum(valid) / len(valid), elapsed


def wait_for_server(base_url: str, timeout: int = 600) -> bool:
    logger.info("Waiting for %s ...", base_url)
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = httpx.get(f"{base_url}/health", timeout=5.0)
            if resp.status_code == 200:
                logger.info("Server %s ready (%.1fs)", base_url, time.time() - start)
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


async def compute_all_losses(
    base_url: str, model: str, sequences: list[list[int]], concurrency: int,
) -> tuple[list[float], float]:
    """Compute losses for all sequences with async concurrency. Returns (losses, total_time)."""
    semaphore = asyncio.Semaphore(concurrency)
    client = httpx.AsyncClient(timeout=120.0)
    completed = 0
    total = len(sequences)

    async def run_one(idx):
        nonlocal completed
        async with semaphore:
            loss, _ = await compute_loss_async(client, base_url, model, sequences[idx])
            completed += 1
            if completed % 20 == 0 or completed == total:
                logger.info("  %s: %d/%d", model.split("/")[-1], completed, total)
            return loss

    t0 = time.time()
    tasks = [run_one(i) for i in range(total)]
    losses = await asyncio.gather(*tasks)
    elapsed = time.time() - t0
    await client.aclose()
    return list(losses), elapsed


def run_sign_test(king_losses: list[float], challenger_losses: list[float], alpha: float) -> dict:
    """Run the sign test on pre-computed paired losses."""
    N = len(king_losses)
    K = int(binom.isf(alpha, N, 0.5))

    s = 0
    n = 0
    n_ties = 0
    king_loss_sum = 0.0
    challenger_loss_sum = 0.0
    early_stop_reason = "full_eval"

    for i in range(N):
        kl = king_losses[i]
        cl = challenger_losses[i]
        king_loss_sum += kl
        challenger_loss_sum += cl

        if kl == cl:
            n_ties += 1
        else:
            n += 1
            if cl < kl:
                s += 1

        if n > 0:
            if s >= K:
                early_stop_reason = "challenger_reached_K"
                break
            remaining = N - (n + n_ties)
            if s + remaining < K:
                early_stop_reason = "king_unreachable"
                break

    total_evaluated = n + n_ties
    accepted = s >= K

    return {
        "accepted": accepted,
        "verdict": "challenger" if accepted else "king",
        "S_N": s,
        "K": K,
        "N": N,
        "n_evaluated": n,
        "n_ties": n_ties,
        "total_sequences_processed": total_evaluated,
        "win_rate": round(s / n if n > 0 else 0, 6),
        "alpha": alpha,
        "early_stopped": early_stop_reason != "full_eval",
        "early_stop_reason": early_stop_reason,
        "avg_king_loss": round(king_loss_sum / total_evaluated, 6) if total_evaluated > 0 else 0,
        "avg_challenger_loss": round(challenger_loss_sum / total_evaluated, 6) if total_evaluated > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Optimized mock duel via vLLM (async batched)")
    parser.add_argument("--king-url", required=True)
    parser.add_argument("--challenger-url", required=True)
    parser.add_argument("--king-model", default="unconst/Teutonic-I")
    parser.add_argument("--challenger-model", default="unconst/Teutonic-I-challenger")
    parser.add_argument("--dataset", default="/tmp/teutonic/synthetic_eval.npy")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--N", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    if not wait_for_server(args.king_url):
        return
    if not wait_for_server(args.challenger_url):
        return

    tokens = np.load(args.dataset, allow_pickle=False).reshape(-1)
    n_sequences = len(tokens) // args.seq_len
    actual_N = min(args.N, n_sequences)
    sequences = [tokens[i * args.seq_len:(i + 1) * args.seq_len].tolist() for i in range(actual_N)]
    logger.info("Loaded %d sequences of length %d", actual_N, args.seq_len)

    t_total = time.time()

    # Compute king losses
    logger.info("Computing king losses (concurrency=%d)...", args.concurrency)
    king_losses, king_time = asyncio.run(
        compute_all_losses(args.king_url, args.king_model, sequences, args.concurrency)
    )
    logger.info("King: %.1f seq/s (%.2fs)", actual_N / king_time, king_time)

    # Compute challenger losses
    logger.info("Computing challenger losses (concurrency=%d)...", args.concurrency)
    challenger_losses, challenger_time = asyncio.run(
        compute_all_losses(args.challenger_url, args.challenger_model, sequences, args.concurrency)
    )
    logger.info("Challenger: %.1f seq/s (%.2fs)", actual_N / challenger_time, challenger_time)

    # Run sign test on pre-computed losses
    verdict = run_sign_test(king_losses, challenger_losses, args.alpha)
    total_time = time.time() - t_total

    verdict["wall_time_s"] = round(total_time, 2)
    verdict["king_inference_time_s"] = round(king_time, 2)
    verdict["challenger_inference_time_s"] = round(challenger_time, 2)
    verdict["total_sequences_per_second"] = round(2 * actual_N / total_time, 3)
    verdict["concurrency"] = args.concurrency

    print("\n" + "=" * 70)
    print("FAST MOCK DUEL VERDICT")
    print("=" * 70)
    print(json.dumps(verdict, indent=2))
    print("=" * 70)

    with open("/tmp/teutonic/duel_verdict.json", "w") as f:
        json.dump(verdict, f, indent=2)
    print("Saved to /tmp/teutonic/duel_verdict.json")


if __name__ == "__main__":
    main()
