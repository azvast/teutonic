#!/bin/bash
pkill -f "vllm serve" 2>/dev/null || true
sleep 3

echo "Starting king (0.40 gpu util)..."
nohup vllm serve /tmp/teutonic/king \
    --port 8000 \
    --dtype bfloat16 \
    --max-model-len 2560 \
    --gpu-memory-utilization 0.40 \
    --disable-log-requests \
    --served-model-name "unconst/Teutonic-I" \
    > /tmp/teutonic/king_server.log 2>&1 &
echo "King PID: $!"

# Wait for king to finish loading before starting challenger to avoid GPU contention
echo "Waiting for king to load first..."
for i in $(seq 1 60); do
    curl -sf http://localhost:8000/health > /dev/null 2>&1 && echo "King ready!" && break
    sleep 2
done

echo "Starting challenger (0.40 gpu util)..."
nohup vllm serve /tmp/teutonic/challenger \
    --port 8001 \
    --dtype bfloat16 \
    --max-model-len 2560 \
    --gpu-memory-utilization 0.40 \
    --disable-log-requests \
    --served-model-name "unconst/Teutonic-I-challenger" \
    > /tmp/teutonic/challenger_server.log 2>&1 &
echo "Challenger PID: $!"

echo "Waiting for both servers..."
for i in $(seq 1 90); do
    k=0; c=0
    curl -sf http://localhost:8000/health > /dev/null 2>&1 && k=1
    curl -sf http://localhost:8001/health > /dev/null 2>&1 && c=1
    
    if [ $k -eq 1 ] && [ $c -eq 1 ]; then
        echo "Both ready after $((i*2)) seconds!"
        nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
        exit 0
    fi
    
    [ $((i % 10)) -eq 0 ] && echo "Waiting... king=$k chal=$c (${i}x2s)"
    sleep 2
done

echo "FAILED"
echo "=== King log ==="
tail -15 /tmp/teutonic/king_server.log
echo "=== Challenger log ==="
tail -15 /tmp/teutonic/challenger_server.log
exit 1
