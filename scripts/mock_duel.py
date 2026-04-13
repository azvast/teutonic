#!/usr/bin/env python3
"""Mock duel: batch sign test between king and challenger via vLLM logprobs.

Sends eval sequences in parallel batches to two vLLM servers,
computes cross-entropy loss from prompt_logprobs, and runs the sign test.
"""

import argparse
import asyncio
import json
import logging
import time

import httpx
import numpy as np
from scipy.stats import binom

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
logger = logging.getLogger("mock_duel")

BATCH_SIZE = 8


async def score_batch(client: httpx.AsyncClient, base_url: str, model: str,
                      token_batches: list[list[int]]) -> list[float | None]:
    """Score multiple sequences concurrently against a vLLM server."""
    async def _score_one(tokens: list[int]) -> float | None:
        try:
            resp = await client.post(
                f"{base_url}/v1/completions",
                json={
                    "model": model,
                    "prompt": tokens,
                    "max_tokens": 1,
                    "echo": True,
                    "logprobs": 1,
                    "temperature": 0.0,
                },
                timeout=120.0,
            )
            resp.raise_for_status()
            data = resp.json()
            lps = data["choices"][0]["logprobs"]["token_logprobs"]
            valid = [lp for lp in lps[1:] if lp is not None]
            if not valid:
                return None
            return -sum(valid) / len(valid)
        except Exception as e:
            logger.error("Score failed: %s", e)
            return None

    return await asyncio.gather(*[_score_one(t) for t in token_batches])


async def wait_for_server(base_url: str, timeout: int = 900) -> bool:
    """Wait for vLLM server to be ready."""
    logger.info("Waiting for %s ...", base_url)
    start = time.time()
    async with httpx.AsyncClient() as client:
        while time.time() - start < timeout:
            try:
                resp = await client.get(f"{base_url}/health", timeout=5.0)
                if resp.status_code == 200:
                    logger.info("Server %s ready!", base_url)
                    return True
            except Exception:
                pass
            await asyncio.sleep(5)
    logger.error("Server %s not ready after %ds", base_url, timeout)
    return False


async def run_duel(
    king_url: str,
    challenger_url: str,
    king_model: str,
    challenger_model: str,
    dataset_path: str,
    seq_len: int,
    N: int,
    alpha: float,
    batch_size: int,
):
    if not await wait_for_server(king_url):
        return None
    if not await wait_for_server(challenger_url):
        return None

    logger.info("Loading dataset from %s", dataset_path)
    tokens = np.load(dataset_path, allow_pickle=False)
    if tokens.dtype != np.uint32:
        tokens = tokens.astype(np.uint32, copy=False)
    tokens = tokens.reshape(-1)
    n_sequences = len(tokens) // seq_len
    actual_N = min(N, n_sequences)

    K = int(binom.isf(alpha, actual_N, 0.5))
    logger.info("Sign test: N=%d, K=%d, alpha=%s, batch_size=%d", actual_N, K, alpha, batch_size)

    rng = np.random.default_rng(42)
    indices = rng.choice(n_sequences, size=actual_N, replace=False)

    s = 0
    n = 0
    n_ties = 0
    king_loss_sum = 0.0
    challenger_loss_sum = 0.0
    t0 = time.time()
    early_stopped = False

    async with httpx.AsyncClient() as king_client, httpx.AsyncClient() as chall_client:
        for batch_start in range(0, actual_N, batch_size):
            batch_indices = indices[batch_start : batch_start + batch_size]
            batch_seqs = [
                tokens[int(idx) * seq_len : (int(idx) + 1) * seq_len].tolist()
                for idx in batch_indices
            ]

            king_losses, chall_losses = await asyncio.gather(
                score_batch(king_client, king_url, king_model, batch_seqs),
                score_batch(chall_client, challenger_url, challenger_model, batch_seqs),
            )

            for kl, cl in zip(king_losses, chall_losses):
                if kl is None or cl is None:
                    continue

                king_loss_sum += kl
                challenger_loss_sum += cl

                if kl == cl:
                    n_ties += 1
                else:
                    n += 1
                    if cl < kl:
                        s += 1

                    if s >= K:
                        logger.info("EARLY STOP: challenger wins (s=%d >= K=%d)", s, K)
                        early_stopped = True
                        break
                    remaining = actual_N - (n + n_ties)
                    if remaining > 0 and s + remaining < K:
                        logger.info("EARLY STOP: king wins (s=%d, can't reach K=%d)", s, K)
                        early_stopped = True
                        break

            if early_stopped:
                break

            evaluated = n + n_ties
            if evaluated > 0 and (batch_start + batch_size) % (batch_size * 2) == 0:
                wr = s / n if n > 0 else 0.0
                elapsed = time.time() - t0
                seqs_per_sec = evaluated / elapsed
                logger.info(
                    "Progress: %d/%d | s=%d n=%d wr=%.3f | king=%.3f chall=%.3f | %.1f seq/s",
                    evaluated, actual_N, s, n, wr,
                    king_loss_sum / evaluated, challenger_loss_sum / evaluated,
                    seqs_per_sec,
                )

    elapsed = time.time() - t0
    total_evaluated = n + n_ties
    accepted = s >= K
    win_rate = s / n if n > 0 else 0.0

    verdict = {
        "accepted": accepted,
        "verdict": "challenger" if accepted else "king",
        "S_N": s,
        "K": K,
        "N": actual_N,
        "n_evaluated": n,
        "n_ties": n_ties,
        "win_rate": round(win_rate, 6),
        "alpha": alpha,
        "avg_king_loss": round(king_loss_sum / total_evaluated, 6) if total_evaluated > 0 else 0,
        "avg_challenger_loss": round(challenger_loss_sum / total_evaluated, 6) if total_evaluated > 0 else 0,
        "wall_time_s": round(elapsed, 1),
        "seqs_per_sec": round(total_evaluated / elapsed, 2) if elapsed > 0 else 0,
        "early_stopped": early_stopped,
    }

    print("\n" + "=" * 60)
    print("MOCK DUEL VERDICT")
    print("=" * 60)
    print(json.dumps(verdict, indent=2))
    print("=" * 60)

    return verdict


def main():
    parser = argparse.ArgumentParser(description="Mock duel via vLLM logprobs")
    parser.add_argument("--king-url", required=True)
    parser.add_argument("--challenger-url", required=True)
    parser.add_argument("--king-model", default="unconst/Teutonic-I")
    parser.add_argument("--challenger-model", default="unconst/Teutonic-I-challenger")
    parser.add_argument("--dataset", default="/tmp/teutonic/synthetic_eval.npy")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--N", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    asyncio.run(run_duel(
        king_url=args.king_url,
        challenger_url=args.challenger_url,
        king_model=args.king_model,
        challenger_model=args.challenger_model,
        dataset_path=args.dataset,
        seq_len=args.seq_len,
        N=args.N,
        alpha=args.alpha,
        batch_size=args.batch_size,
    ))


if __name__ == "__main__":
    main()
