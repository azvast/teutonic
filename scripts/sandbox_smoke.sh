#!/usr/bin/env bash
# Sandbox smoke-test driver for the Teutonic-LXXX 80B Qwen3MoE soak.
# Run on the sandbox box AFTER mock-king + mock-chall both exist on HF.
#
# 1. Brings up eval_server in sharded mode in tmux pane "eval".
# 2. POSTs /eval, captures the eval_id (the API is async).
# 3. Streams /eval/{id}/stream until the verdict (or failure) event arrives.
# 4. Records nvidia-smi dmon for the duration.
# 5. Writes everything to /workspace/logs/smoke-<ts>/.
set -euo pipefail

EVAL_N=${EVAL_N:-2000}
BATCH=${BATCH:-32}
BOOT=${BOOT:-2000}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --eval-n) EVAL_N=$2; shift 2;;
    --batch-size) BATCH=$2; shift 2;;
    --bootstrap) BOOT=$2; shift 2;;
    *) echo "unknown arg $1"; exit 2;;
  esac
done

TS=$(date -u +%Y%m%dT%H%M%SZ)
OUT=/workspace/logs/smoke-${TS}
mkdir -p "$OUT"
echo "smoke-test output dir: $OUT (eval_n=$EVAL_N batch=$BATCH bootstrap=$BOOT)"

cd /root/teutonic
source .venv/bin/activate
source /root/.creds/hf_token.env

# 1. Bring eval_server up in sharded mode (tmux session keeps it alive).
echo "=== killing any prior eval-server tmux session ==="
tmux kill-session -t eval 2>/dev/null || true
sleep 2

echo "=== launching eval_server in tmux 'eval' ==="
tmux new -d -s eval "
  cd /root/teutonic
  source .venv/bin/activate
  source /root/.creds/hf_token.env
  TEUTONIC_SHARD_ACROSS_GPUS=1 \
  TEUTONIC_PROBE_ENABLED=0 \
  EVAL_N=$EVAL_N \
  EVAL_BATCH_SIZE=$BATCH \
  EVAL_BOOTSTRAP_B=$BOOT \
  HF_CACHE_HIGH_WATERMARK_GB=600 \
  HF_PREFETCH_TIMEOUT=1800 \
  EVAL_MAX_RUNTIME_S=7200 \
  uvicorn eval_server:app --host 127.0.0.1 --port 9000 \
    > $OUT/eval-server.log 2>&1
"

# 2. Wait for /health.
echo "=== waiting for /health ==="
for i in $(seq 1 60); do
  if curl -sf http://127.0.0.1:9000/health >/dev/null 2>&1; then
    echo "eval_server up after ${i}s"
    break
  fi
  sleep 1
done

# 3. nvidia-smi dmon in background (per-GPU memory + util every 5s).
nvidia-smi dmon -s mu -i 0,1,2,3,4,5,6,7 -d 5 \
  > "$OUT/nvidia-smi-dmon.log" 2>&1 &
DMON_PID=$!
echo "nvidia-smi dmon PID=$DMON_PID"

# 4. POST /eval — async, returns eval_id immediately.
echo "=== POST /eval ==="
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$OUT/eval-start.ts"
RESP=$(curl -sS -X POST http://127.0.0.1:9000/eval \
  -H 'content-type: application/json' \
  -d "{\"king_repo\":\"unconst/Teutonic-LXXX-mock-king\",\"challenger_repo\":\"unconst/Teutonic-LXXX-mock-chall\",\"block_hash\":\"smoke\",\"hotkey\":\"smoke\",\"shard_key\":\"dataset/lxxx-smoke/shard_smoke.npy\",\"eval_n\":$EVAL_N,\"alpha\":0.001,\"seq_len\":2048,\"batch_size\":$BATCH,\"n_bootstrap\":$BOOT}")
echo "POST response: $RESP"
EVAL_ID=$(printf '%s' "$RESP" | python3 -c 'import sys,json; print(json.load(sys.stdin)["eval_id"])')
echo "eval_id=$EVAL_ID"
echo "$EVAL_ID" > "$OUT/eval-id"

# 5. Stream SSE in background; poll the JSON status until completed/failed.
echo "=== streaming /eval/$EVAL_ID/stream ==="
curl -sN -m 7200 "http://127.0.0.1:9000/eval/$EVAL_ID/stream" \
  > "$OUT/sse.log" 2>&1 &
CURL_PID=$!

for i in $(seq 1 1440); do
  sleep 5
  STATUS=$(curl -sS "http://127.0.0.1:9000/eval/$EVAL_ID" 2>/dev/null || echo '{}')
  STATE=$(printf '%s' "$STATUS" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("state","?"))' 2>/dev/null || echo '?')
  PHASE=$(printf '%s' "$STATUS" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("stage","-"))' 2>/dev/null || echo '-')
  ELAPSED=$(printf '%s' "$STATUS" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(int(d.get("stage_elapsed_s",0)))' 2>/dev/null || echo '0')
  printf '[%s] poll %3d  state=%s  stage=%s  elapsed=%ss\n' "$(date +%H:%M:%S)" "$i" "$STATE" "$PHASE" "$ELAPSED" | tee -a "$OUT/poll.log"
  if [ "$STATE" = "completed" ] || [ "$STATE" = "failed" ]; then
    break
  fi
done

date -u +'%Y-%m-%dT%H:%M:%SZ' > "$OUT/eval-end.ts"
kill $CURL_PID 2>/dev/null || true
kill $DMON_PID 2>/dev/null || true
sleep 2

# 6. Final status snapshot.
curl -sS "http://127.0.0.1:9000/eval/$EVAL_ID" \
  | python3 -m json.tool > "$OUT/final-status.json" 2>/dev/null || true

# 7. Summary.
echo ""
echo "=== summary ==="
echo "start: $(cat $OUT/eval-start.ts)"
echo "end:   $(cat $OUT/eval-end.ts)"
echo ""
echo "final status (truncated):"
head -c 4000 "$OUT/final-status.json"
echo
echo ""
echo "peak per-GPU mem_used (MiB):"
awk '/^[ ]*[0-9]/ { if (NF >= 5) { gpu=$1; mem=$NF; if (mem+0 > peak[gpu]) peak[gpu]=mem+0 } } END { for (g in peak) print "  GPU "g": "peak[g]" MiB" }' "$OUT/nvidia-smi-dmon.log" | sort
echo ""
echo "report dir: $OUT"
echo ""
echo "(eval_server is still alive in tmux 'eval'; tmux kill-session -t eval to stop)"
