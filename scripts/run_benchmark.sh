#!/bin/bash
set -e

echo "=== Starting single vLLM server for king ==="
nohup vllm serve /tmp/teutonic/king \
    --port 8000 \
    --dtype bfloat16 \
    --max-model-len 2560 \
    --gpu-memory-utilization 0.85 \
    --disable-log-requests \
    --served-model-name "unconst/Teutonic-I" \
    > /tmp/teutonic/king_server.log 2>&1 &
KING_PID=$!
echo "King PID: $KING_PID"

echo "Waiting for server..."
for i in $(seq 1 60); do
    curl -sf http://localhost:8000/health > /dev/null 2>&1 && echo "Ready after $((i*2))s!" && break
    [ $((i % 10)) -eq 0 ] && echo "Still waiting... (${i}x2s)"
    sleep 2
done

nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
echo ""

echo "=== Running Benchmark ==="
python3 /tmp/teutonic/benchmark_vllm.py \
    --url http://localhost:8000 \
    --model "unconst/Teutonic-I" \
    --dataset /tmp/teutonic/synthetic_eval.npy \
    --n 50 \
    --concurrency-levels "1,2,4,8,16"

echo ""
echo "=== Benchmark Complete ==="
kill $KING_PID 2>/dev/null || true
