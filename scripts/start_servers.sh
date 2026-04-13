#!/bin/bash
pkill -f "vllm serve" 2>/dev/null || true
sleep 2

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

echo "Servers starting in background"
