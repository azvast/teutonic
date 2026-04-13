#!/bin/bash
pkill -f "vllm serve" 2>/dev/null || true
sleep 3

nohup vllm serve /tmp/teutonic/king \
    --port 8000 \
    --dtype bfloat16 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.45 \
    --disable-log-requests \
    --served-model-name "unconst/Teutonic-I" \
    > /tmp/teutonic/king_server.log 2>&1 &
echo "King PID: $!"

nohup vllm serve /tmp/teutonic/challenger \
    --port 8001 \
    --dtype bfloat16 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.45 \
    --disable-log-requests \
    --served-model-name "unconst/Teutonic-I-challenger" \
    > /tmp/teutonic/challenger_server.log 2>&1 &
echo "Challenger PID: $!"

echo "Waiting for servers..."
for i in $(seq 1 90); do
    king_ok=0
    chal_ok=0
    curl -sf http://localhost:8000/health > /dev/null 2>&1 && king_ok=1
    curl -sf http://localhost:8001/health > /dev/null 2>&1 && chal_ok=1
    
    if [ $king_ok -eq 1 ] && [ $chal_ok -eq 1 ]; then
        echo "Both servers ready after ${i}x2 seconds!"
        nvidia-smi
        exit 0
    fi
    
    if [ $((i % 10)) -eq 0 ]; then
        echo "Still waiting... king=$king_ok challenger=$chal_ok (${i}x2s)"
    fi
    sleep 2
done

echo "ERROR: Servers did not start in time"
echo "=== King log ==="
tail -20 /tmp/teutonic/king_server.log
echo "=== Challenger log ==="
tail -20 /tmp/teutonic/challenger_server.log
exit 1
