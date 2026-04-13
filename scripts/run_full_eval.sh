#!/bin/bash
set -e

KING_DIR="/tmp/teutonic/king"
CHALLENGER_DIR="/tmp/teutonic/challenger"
DATASET="/tmp/teutonic/synthetic_eval.npy"
KING_PORT=8000
CHALLENGER_PORT=8001

echo "=== Starting vLLM servers ==="

# Kill any existing vLLM processes
pkill -f "vllm serve" 2>/dev/null || true
sleep 2

# Start king server (background)
echo "Starting king server on port $KING_PORT..."
vllm serve "$KING_DIR" \
    --port $KING_PORT \
    --dtype bfloat16 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.45 \
    --disable-log-requests \
    --served-model-name "unconst/Teutonic-I" \
    > /tmp/teutonic/king_server.log 2>&1 &
KING_PID=$!
echo "King PID: $KING_PID"

# Start challenger server (background)
echo "Starting challenger server on port $CHALLENGER_PORT..."
vllm serve "$CHALLENGER_DIR" \
    --port $CHALLENGER_PORT \
    --dtype bfloat16 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.45 \
    --disable-log-requests \
    --served-model-name "unconst/Teutonic-I-challenger" \
    > /tmp/teutonic/challenger_server.log 2>&1 &
CHALLENGER_PID=$!
echo "Challenger PID: $CHALLENGER_PID"

echo "Waiting for servers to start..."

# Wait for both servers
for port in $KING_PORT $CHALLENGER_PORT; do
    url="http://localhost:$port"
    for i in $(seq 1 120); do
        if curl -sf "$url/health" > /dev/null 2>&1; then
            echo "Server on port $port is ready!"
            break
        fi
        if [ $i -eq 120 ]; then
            echo "ERROR: Server on port $port failed to start"
            echo "=== King server log ==="
            tail -30 /tmp/teutonic/king_server.log
            echo "=== Challenger server log ==="
            tail -30 /tmp/teutonic/challenger_server.log
            kill $KING_PID $CHALLENGER_PID 2>/dev/null
            exit 1
        fi
        sleep 2
    done
done

echo ""
echo "=== Both servers running ==="
echo ""

# Run benchmark on king model
echo "=== Running Benchmark ==="
python3 /tmp/teutonic/benchmark_vllm.py \
    --url "http://localhost:$KING_PORT" \
    --model "unconst/Teutonic-I" \
    --dataset "$DATASET" \
    --n 50 \
    --concurrency-levels "1,2,4,8,16"

echo ""

# Run the optimized mock duel
echo "=== Running Fast Mock Duel ==="
python3 /tmp/teutonic/mock_duel_fast.py \
    --king-url "http://localhost:$KING_PORT" \
    --challenger-url "http://localhost:$CHALLENGER_PORT" \
    --king-model "unconst/Teutonic-I" \
    --challenger-model "unconst/Teutonic-I-challenger" \
    --dataset "$DATASET" \
    --N 100 \
    --alpha 0.05 \
    --concurrency 8

echo ""

# Also run the original mock duel for comparison
echo "=== Running Original Mock Duel (for comparison) ==="
python3 /tmp/teutonic/mock_duel.py \
    --king-url "http://localhost:$KING_PORT" \
    --challenger-url "http://localhost:$CHALLENGER_PORT" \
    --king-model "unconst/Teutonic-I" \
    --challenger-model "unconst/Teutonic-I-challenger" \
    --dataset "$DATASET" \
    --N 100 \
    --alpha 0.05

echo ""
echo "=== All done ==="

# Cleanup
kill $KING_PID $CHALLENGER_PID 2>/dev/null || true
