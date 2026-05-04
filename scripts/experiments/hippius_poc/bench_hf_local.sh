#!/usr/bin/env bash
# Local-box HF download benchmark. Targets the 46GiB Quasar king shard.
#
# Variants:
#   baseline_w8     snapshot_download max_workers=8 (HF default)
#   baseline_w16    snapshot_download max_workers=16
#   hf_transfer_w4  HF_HUB_ENABLE_HF_TRANSFER=1, max_workers=4
#   hf_transfer_w8  HF_HUB_ENABLE_HF_TRANSFER=1, max_workers=8
#   hf_transfer_w16 HF_HUB_ENABLE_HF_TRANSFER=1, max_workers=16
#   aria2c_x16      raw aria2c with 16 connections, 16 splits
#
# Usage:
#   bash bench_hf_local.sh                       # all variants
#   bash bench_hf_local.sh baseline_w8 hf_transfer_w8

set -euo pipefail

REPO="${REPO:-unconst/Teutonic-XXIV}"
SHARD="${SHARD:-model-00001-of-00002.safetensors}"
CACHE_ROOT="${CACHE_ROOT:-/tmp/hfbench}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="$(cd "$(dirname "$0")" && pwd)/runs/hf_local-${RUN_ID}"
mkdir -p "${RUN_DIR}"
JSONL="${RUN_DIR}/results.jsonl"

PY="/home/const/workspace/.venv/bin/python"
HF_TOKEN_VAL="$(doppler secrets get HF_TOKEN --plain -p arbos -c prd)"

echo "[bench] target ${REPO}/${SHARD} -> ${CACHE_ROOT}/<variant>/"
echo "[bench] log ${JSONL}"

free_disk() {
  df -B1 /tmp | awk 'NR==2 {print $4}'
}

drop_caches() {
  # best effort; needs root for full effect. Skip silently.
  sudo -n sysctl vm.drop_caches=3 >/dev/null 2>&1 || true
}

emit() {
  local variant="$1" bytes="$2" seconds="$3" extra="$4"
  local mbps
  mbps="$(awk -v b="${bytes}" -v s="${seconds}" 'BEGIN{printf "%.2f", (b/1024/1024)/s}')"
  printf '{"variant":"%s","bytes":%s,"seconds":%.3f,"mbps":%s,"extra":%s}\n' \
    "${variant}" "${bytes}" "${seconds}" "${mbps}" "${extra}" >> "${JSONL}"
  echo "[bench] variant=${variant} bytes=${bytes} seconds=${seconds} mbps=${mbps}"
}

run_snapshot() {
  local variant="$1" workers="$2" use_hf_transfer="$3"
  local cache="${CACHE_ROOT}/${variant}"
  rm -rf "${cache}"; mkdir -p "${cache}"
  drop_caches
  local t0=$(date +%s.%N)
  HF_HUB_ENABLE_HF_TRANSFER="${use_hf_transfer}" \
  HF_TOKEN="${HF_TOKEN_VAL}" \
  "${PY}" - <<PY
import os, time
from huggingface_hub import snapshot_download
t0 = time.monotonic()
local = snapshot_download(
    repo_id="${REPO}",
    local_dir="${cache}",
    token=os.environ.get("HF_TOKEN") or None,
    allow_patterns=["${SHARD}"],
    max_workers=${workers},
)
print("done", time.monotonic() - t0)
PY
  local t1=$(date +%s.%N)
  local bytes
  bytes=$(stat -c%s "${cache}/${SHARD}")
  local seconds
  seconds=$(awk -v a="${t0}" -v b="${t1}" 'BEGIN{printf "%.3f", b-a}')
  emit "${variant}" "${bytes}" "${seconds}" "{\"workers\":${workers},\"hf_transfer\":${use_hf_transfer:-0}}"
  rm -rf "${cache}"
}

run_aria2c() {
  local variant="aria2c_x16"
  local cache="${CACHE_ROOT}/${variant}"
  rm -rf "${cache}"; mkdir -p "${cache}"
  drop_caches
  local url="https://huggingface.co/${REPO}/resolve/main/${SHARD}"
  local t0=$(date +%s.%N)
  aria2c \
    --header="Authorization: Bearer ${HF_TOKEN_VAL}" \
    -x 16 -s 16 -k 64M \
    --check-integrity=false \
    --console-log-level=warn \
    --summary-interval=5 \
    -d "${cache}" -o "${SHARD}" \
    "${url}"
  local t1=$(date +%s.%N)
  local bytes
  bytes=$(stat -c%s "${cache}/${SHARD}")
  local seconds
  seconds=$(awk -v a="${t0}" -v b="${t1}" 'BEGIN{printf "%.3f", b-a}')
  emit "${variant}" "${bytes}" "${seconds}" "{\"connections\":16,\"split\":16,\"chunk\":\"64M\"}"
  rm -rf "${cache}"
}

VARIANTS=("$@")
if [[ ${#VARIANTS[@]} -eq 0 ]]; then
  VARIANTS=(baseline_w8 baseline_w16 hf_transfer_w4 hf_transfer_w8 hf_transfer_w16 aria2c_x16)
fi

for v in "${VARIANTS[@]}"; do
  echo
  echo "==================================================="
  echo "[bench] starting variant=${v}"
  echo "==================================================="
  case "${v}" in
    baseline_w8)      run_snapshot "${v}" 8  0 ;;
    baseline_w16)     run_snapshot "${v}" 16 0 ;;
    hf_transfer_w4)   run_snapshot "${v}" 4  1 ;;
    hf_transfer_w8)   run_snapshot "${v}" 8  1 ;;
    hf_transfer_w16)  run_snapshot "${v}" 16 1 ;;
    aria2c_x16)       run_aria2c ;;
    *) echo "unknown variant ${v}" >&2; exit 1 ;;
  esac
done

echo
echo "[bench] summary:"
column -t -s'	' < <(python3 -c "
import json,sys
rows=[json.loads(l) for l in open('${JSONL}')]
print('variant\tbytes\tseconds\tmbps\textra')
for r in rows: print(r['variant'],r['bytes'],r['seconds'],r['mbps'],json.dumps(r['extra']),sep='\t')
")
