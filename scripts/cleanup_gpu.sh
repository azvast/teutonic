#!/bin/bash
# Kill all vllm and stale python processes, then check GPU
for pid in $(pgrep -f "vllm serve" 2>/dev/null); do
    kill -9 $pid 2>/dev/null
done
sleep 3
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader
echo "GPU cleaned"
