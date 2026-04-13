#!/usr/bin/env python3
"""Benchmark vLLM inference speed: single vs batched, various batch sizes.

Measures throughput in sequences/second and tokens/second for the Teutonic-I
model using vLLM's /v1/completions endpoint with prompt_logprobs.
"""

import argparse
import asyncio
import json
import logging
import time

import httpx
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
logger = logging.getLogger("benchmark")


def wait_for_server(base_url: str, timeout: int = 600) -> bool:
    logger.info("Waiting for %s ...", base_url)
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = httpx.get(f"{base_url}/health", timeout=5.0)
            if resp.status_code == 200:
                logger.info("Server ready (%.1fs)", time.time() - start)
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def compute_loss_sync(client: httpx.Client, base_url: str, model: str, token_ids: list[int]) -> float:
    resp = client.post(
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
    data = resp.json()
    logprobs_list = data["choices"][0]["logprobs"]["token_logprobs"]
    # Skip the first (None for BOS) and the last (generated token)
    valid = [lp for lp in logprobs_list[1:-1] if lp is not None]
    if not valid:
        return float("nan")
    return -sum(valid) / len(valid)


async def compute_loss_async(client: httpx.AsyncClient, base_url: str, model: str, token_ids: list[int]) -> float:
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
    data = resp.json()
    logprobs_list = data["choices"][0]["logprobs"]["token_logprobs"]
    valid = [lp for lp in logprobs_list[1:-1] if lp is not None]
    if not valid:
        return float("nan")
    return -sum(valid) / len(valid)


def benchmark_sequential(base_url: str, model: str, sequences: list, n: int):
    """Benchmark: send one request at a time."""
    logger.info("=== Sequential benchmark: %d sequences ===", n)
    client = httpx.Client()
    losses = []
    t0 = time.time()
    for i in range(n):
        loss = compute_loss_sync(client, base_url, model, sequences[i])
        losses.append(loss)
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            logger.info("  %d/%d done (%.1f seq/s)", i + 1, n, (i + 1) / elapsed)
    elapsed = time.time() - t0
    client.close()
    seq_len = len(sequences[0])
    return {
        "mode": "sequential",
        "n_sequences": n,
        "seq_len": seq_len,
        "elapsed_s": round(elapsed, 2),
        "seq_per_s": round(n / elapsed, 3),
        "tokens_per_s": round(n * seq_len / elapsed, 1),
        "avg_loss": round(np.mean(losses), 6),
    }


async def benchmark_concurrent(base_url: str, model: str, sequences: list, n: int, concurrency: int):
    """Benchmark: send multiple requests concurrently."""
    logger.info("=== Concurrent benchmark: %d sequences, concurrency=%d ===", n, concurrency)
    semaphore = asyncio.Semaphore(concurrency)
    client = httpx.AsyncClient()
    losses = []
    completed = 0

    async def run_one(idx):
        nonlocal completed
        async with semaphore:
            loss = await compute_loss_async(client, base_url, model, sequences[idx])
            completed += 1
            return loss

    t0 = time.time()
    tasks = [run_one(i) for i in range(n)]
    losses = await asyncio.gather(*tasks)
    elapsed = time.time() - t0
    await client.aclose()

    seq_len = len(sequences[0])
    return {
        "mode": f"concurrent_{concurrency}",
        "n_sequences": n,
        "seq_len": seq_len,
        "concurrency": concurrency,
        "elapsed_s": round(elapsed, 2),
        "seq_per_s": round(n / elapsed, 3),
        "tokens_per_s": round(n * seq_len / elapsed, 1),
        "avg_loss": round(float(np.mean(losses)), 6),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark vLLM inference speed")
    parser.add_argument("--url", required=True, help="vLLM server URL")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--dataset", default="/tmp/teutonic/synthetic_eval.npy")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--n", type=int, default=50, help="Sequences to benchmark")
    parser.add_argument("--concurrency-levels", default="1,2,4,8,16,32", help="Concurrency levels to test")
    args = parser.parse_args()

    if not wait_for_server(args.url):
        logger.error("Server not ready")
        return

    tokens = np.load(args.dataset, allow_pickle=False).reshape(-1)
    n_sequences = len(tokens) // args.seq_len
    n = min(args.n, n_sequences)
    sequences = [tokens[i * args.seq_len:(i + 1) * args.seq_len].tolist() for i in range(n)]
    logger.info("Loaded %d sequences of length %d", n, args.seq_len)

    # Warmup
    logger.info("Warmup: 3 sequences...")
    client = httpx.Client()
    for i in range(3):
        compute_loss_sync(client, args.url, args.model, sequences[i])
    client.close()
    logger.info("Warmup done")

    results = []

    # Sequential
    r = benchmark_sequential(args.url, args.model, sequences, n)
    results.append(r)
    logger.info("Sequential: %.3f seq/s, %.1f tok/s", r["seq_per_s"], r["tokens_per_s"])

    # Concurrent at various levels
    for c in [int(x) for x in args.concurrency_levels.split(",")]:
        r = asyncio.run(benchmark_concurrent(args.url, args.model, sequences, n, c))
        results.append(r)
        speedup = r["seq_per_s"] / results[0]["seq_per_s"]
        logger.info(
            "Concurrent(%d): %.3f seq/s, %.1f tok/s (%.1fx speedup)",
            c, r["seq_per_s"], r["tokens_per_s"], speedup,
        )

    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print(f"{'Mode':<20} {'Seq/s':>8} {'Tok/s':>10} {'Time(s)':>8} {'Speedup':>8}")
    print("-" * 80)
    base_rate = results[0]["seq_per_s"]
    for r in results:
        speedup = r["seq_per_s"] / base_rate
        print(f"{r['mode']:<20} {r['seq_per_s']:>8.3f} {r['tokens_per_s']:>10.1f} {r['elapsed_s']:>8.2f} {speedup:>7.1f}x")
    print("=" * 80)

    with open("/tmp/teutonic/benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to /tmp/teutonic/benchmark_results.json")


if __name__ == "__main__":
    main()
