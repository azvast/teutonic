#!/bin/bash
set -e

echo "============================================"
echo " TEUTONIC vLLM EVALUATION PIPELINE"
echo "============================================"
echo ""

# Clean up first
bash /tmp/teutonic/cleanup_gpu.sh || true
sleep 2

# ---- Phase 1: Start king server ----
echo "=== Phase 1: Starting King vLLM Server ==="
nohup vllm serve /tmp/teutonic/king \
    --port 8000 \
    --dtype bfloat16 \
    --max-model-len 2560 \
    --gpu-memory-utilization 0.85 \
    --disable-log-requests \
    --served-model-name "unconst/Teutonic-I" \
    > /tmp/teutonic/king_server.log 2>&1 &
KING_PID=$!

for i in $(seq 1 60); do
    curl -sf http://localhost:8000/health > /dev/null 2>&1 && break
    [ $((i % 10)) -eq 0 ] && echo "  Waiting... (${i}x2s)"
    sleep 2
done
echo "King server ready!"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
echo ""

# ---- Phase 2: Benchmark ----
echo "=== Phase 2: Inference Benchmark ==="
python3 /tmp/teutonic/benchmark_vllm.py \
    --url http://localhost:8000 \
    --model "unconst/Teutonic-I" \
    --dataset /tmp/teutonic/synthetic_eval.npy \
    --n 50 \
    --concurrency-levels "1,2,4,8,16"
echo ""

# ---- Phase 3: Compute king losses for duel ----
echo "=== Phase 3: Computing King Losses (N=100, concurrency=8) ==="
python3 -c "
import asyncio, json, time, numpy as np, httpx

async def compute_loss(client, url, model, tokens):
    resp = await client.post(f'{url}/v1/completions', json={
        'model': model, 'prompt': tokens, 'max_tokens': 1,
        'temperature': 0.0, 'logprobs': 1, 'echo': True,
    }, timeout=120.0)
    resp.raise_for_status()
    lps = resp.json()['choices'][0]['logprobs']['token_logprobs']
    valid = [lp for lp in lps[1:-1] if lp is not None]
    return -sum(valid) / len(valid) if valid else float('nan')

async def main():
    tokens = np.load('/tmp/teutonic/synthetic_eval.npy', allow_pickle=False).reshape(-1)
    N = 100
    seqs = [tokens[i*2048:(i+1)*2048].tolist() for i in range(N)]
    sem = asyncio.Semaphore(8)
    client = httpx.AsyncClient(timeout=120.0)
    done = 0

    async def run(i):
        nonlocal done
        async with sem:
            loss = await compute_loss(client, 'http://localhost:8000', 'unconst/Teutonic-I', seqs[i])
            done += 1
            if done % 20 == 0: print(f'  King: {done}/{N}')
            return loss

    t0 = time.time()
    losses = await asyncio.gather(*[run(i) for i in range(N)])
    elapsed = time.time() - t0
    await client.aclose()
    print(f'King: {N/elapsed:.1f} seq/s ({elapsed:.2f}s)')
    with open('/tmp/teutonic/king_losses.json', 'w') as f:
        json.dump({'losses': list(losses), 'time_s': elapsed}, f)

asyncio.run(main())
"
echo ""

# ---- Phase 4: Swap to challenger server ----
echo "=== Phase 4: Swapping to Challenger Server ==="
kill $KING_PID 2>/dev/null || true
sleep 3
# Wait for GPU memory to free
for i in $(seq 1 10); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)
    [ "$free" -gt 20000 ] && break
    sleep 2
done

nohup vllm serve /tmp/teutonic/challenger \
    --port 8001 \
    --dtype bfloat16 \
    --max-model-len 2560 \
    --gpu-memory-utilization 0.85 \
    --disable-log-requests \
    --served-model-name "unconst/Teutonic-I-challenger" \
    > /tmp/teutonic/challenger_server.log 2>&1 &
CHAL_PID=$!

for i in $(seq 1 60); do
    curl -sf http://localhost:8001/health > /dev/null 2>&1 && break
    [ $((i % 10)) -eq 0 ] && echo "  Waiting... (${i}x2s)"
    sleep 2
done
echo "Challenger server ready!"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
echo ""

# ---- Phase 5: Compute challenger losses ----
echo "=== Phase 5: Computing Challenger Losses (N=100, concurrency=8) ==="
python3 -c "
import asyncio, json, time, numpy as np, httpx

async def compute_loss(client, url, model, tokens):
    resp = await client.post(f'{url}/v1/completions', json={
        'model': model, 'prompt': tokens, 'max_tokens': 1,
        'temperature': 0.0, 'logprobs': 1, 'echo': True,
    }, timeout=120.0)
    resp.raise_for_status()
    lps = resp.json()['choices'][0]['logprobs']['token_logprobs']
    valid = [lp for lp in lps[1:-1] if lp is not None]
    return -sum(valid) / len(valid) if valid else float('nan')

async def main():
    tokens = np.load('/tmp/teutonic/synthetic_eval.npy', allow_pickle=False).reshape(-1)
    N = 100
    seqs = [tokens[i*2048:(i+1)*2048].tolist() for i in range(N)]
    sem = asyncio.Semaphore(8)
    client = httpx.AsyncClient(timeout=120.0)
    done = 0

    async def run(i):
        nonlocal done
        async with sem:
            loss = await compute_loss(client, 'http://localhost:8001', 'unconst/Teutonic-I-challenger', seqs[i])
            done += 1
            if done % 20 == 0: print(f'  Challenger: {done}/{N}')
            return loss

    t0 = time.time()
    losses = await asyncio.gather(*[run(i) for i in range(N)])
    elapsed = time.time() - t0
    await client.aclose()
    print(f'Challenger: {N/elapsed:.1f} seq/s ({elapsed:.2f}s)')
    with open('/tmp/teutonic/challenger_losses.json', 'w') as f:
        json.dump({'losses': list(losses), 'time_s': elapsed}, f)

asyncio.run(main())
"
echo ""

# ---- Phase 6: Run sign test and produce verdict ----
echo "=== Phase 6: Sign Test & Verdict ==="
python3 -c "
import json, time
from scipy.stats import binom

king = json.load(open('/tmp/teutonic/king_losses.json'))
chal = json.load(open('/tmp/teutonic/challenger_losses.json'))

kl = king['losses']
cl = chal['losses']
N = len(kl)
alpha = 0.05
K = int(binom.isf(alpha, N, 0.5))

s = n = n_ties = 0
king_sum = chal_sum = 0.0
reason = 'full_eval'

for i in range(N):
    king_sum += kl[i]
    chal_sum += cl[i]
    if kl[i] == cl[i]:
        n_ties += 1
    else:
        n += 1
        if cl[i] < kl[i]:
            s += 1
    if n > 0:
        if s >= K:
            reason = 'challenger_reached_K'
            break
        rem = N - (n + n_ties)
        if s + rem < K:
            reason = 'king_unreachable'
            break

total = n + n_ties
accepted = s >= K
verdict = {
    'accepted': accepted,
    'verdict': 'challenger' if accepted else 'king',
    'S_N': s, 'K': K, 'N': N,
    'n_evaluated': n, 'n_ties': n_ties,
    'win_rate': round(s/n, 6) if n > 0 else 0,
    'alpha': alpha,
    'early_stopped': reason != 'full_eval',
    'early_stop_reason': reason,
    'avg_king_loss': round(king_sum/total, 6) if total > 0 else 0,
    'avg_challenger_loss': round(chal_sum/total, 6) if total > 0 else 0,
    'king_inference_s': round(king['time_s'], 2),
    'challenger_inference_s': round(chal['time_s'], 2),
    'total_wall_s': round(king['time_s'] + chal['time_s'], 2),
    'throughput_seq_per_s': round(2*N / (king['time_s'] + chal['time_s']), 3),
}

print()
print('=' * 70)
print('MOCK DUEL VERDICT')
print('=' * 70)
print(json.dumps(verdict, indent=2))
print('=' * 70)

with open('/tmp/teutonic/duel_verdict.json', 'w') as f:
    json.dump(verdict, f, indent=2)
"

echo ""
echo "=== Cleanup ==="
kill $CHAL_PID 2>/dev/null || true
echo "Done! Results in /tmp/teutonic/"
