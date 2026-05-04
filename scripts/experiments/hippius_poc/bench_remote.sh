#!/usr/bin/env bash
# Per-tier benchmark sequence -- runnable directly on the B200 box.
#
# Required env (export before invoking, or pass via ssh -o SendEnv):
#   TIER                   tinyllama | quasar
#   HF_REPO                e.g. TinyLlama/TinyLlama-1.1B-Chat-v1.0
#   COLDKEY8               8 hex chars (default: deadbeef)
#   RUN_ID                 e.g. 20260504T125500Z-deadbeef
#   PREFIX_ROOT            poc/${RUN_ID}/challengers
#   BUCKET                 teutonic-sn3
#   ENDPOINT               https://s3.hippius.com
#   TP                     vLLM tensor_parallel_size (default 8)
#   HF_TOKEN               for snapshot_download (optional, public repos work without)
#   TEUTONIC_DS_ACCESS_KEY / TEUTONIC_DS_SECRET_KEY  (or HIPPIUS_*)
#   POC_ROOT               /workspace/hippius_poc (where source lives on the box)
#   CACHE_ROOT             /workspace (parent of hf_cache/, hippius_cache/)

set -euo pipefail

: "${TIER:?need TIER}"
: "${HF_REPO:?need HF_REPO}"
: "${RUN_ID:?need RUN_ID}"
: "${PREFIX_ROOT:?need PREFIX_ROOT}"
: "${BUCKET:=teutonic-sn3}"
: "${ENDPOINT:=https://s3.hippius.com}"
: "${TP:=8}"
: "${COLDKEY8:=deadbeef}"
: "${POC_ROOT:=/workspace/hippius_poc}"
: "${CACHE_ROOT:=/workspace}"
: "${ALLOW_VLLM_FALLBACK:=1}"
: "${KEEP_OBJECTS:=0}"

cd "${POC_ROOT}"
export PYTHONPATH="${POC_ROOT}:${PYTHONPATH:-}"
export TEUTONIC_DS_ENDPOINT="${ENDPOINT}"
export TEUTONIC_DS_BUCKET="${BUCKET}"

RUN_DIR="${POC_ROOT}/runs/${RUN_ID}"
HF_DIR="${CACHE_ROOT}/hf_cache/${TIER}"
DL_DIR="${CACHE_ROOT}/hippius_cache/${TIER}"
mkdir -p "${RUN_DIR}" "${HF_DIR}" "$(dirname "${DL_DIR}")"

PY="${POC_ROOT}/.venv/bin/python"
if [[ ! -x "${PY}" ]]; then
  PY="$(command -v python3 || command -v python)"
fi

echo "[bench_remote] tier=${TIER} repo=${HF_REPO} run_id=${RUN_ID}"
echo "[bench_remote] python=${PY}"
echo "[bench_remote] endpoint=${ENDPOINT} bucket=${BUCKET} prefix=${PREFIX_ROOT}/${COLDKEY8}/<sha256>"

echo "::: stage 1: HF snapshot ::::"
"${PY}" fetch_hf.py \
  --repo "${HF_REPO}" \
  --out "${HF_DIR}" \
  --run-dir "${RUN_DIR}" \
  --tier "${TIER}" \
  --wipe

echo "::: stage 2: Hippius upload ::::"
"${PY}" upload.py \
  --src-dir "${HF_DIR}" \
  --bucket "${BUCKET}" \
  --prefix-root "${PREFIX_ROOT}" \
  --coldkey8 "${COLDKEY8}" \
  --endpoint "${ENDPOINT}" \
  --run-dir "${RUN_DIR}" \
  --tier "${TIER}"

echo "::: stage 3a: Hippius download (positive) ::::"
"${PY}" download.py \
  --commitment "${RUN_DIR}/commitment.${TIER}.json" \
  --bucket "${BUCKET}" \
  --prefix-root "${PREFIX_ROOT}" \
  --out "${DL_DIR}" \
  --endpoint "${ENDPOINT}" \
  --run-dir "${RUN_DIR}" \
  --tier "${TIER}"

echo "::: stage 3b: Hippius download (negative IfMatch) ::::"
"${PY}" download.py \
  --commitment "${RUN_DIR}/commitment.${TIER}.json" \
  --bucket "${BUCKET}" \
  --prefix-root "${PREFIX_ROOT}" \
  --out "${DL_DIR}.neg" \
  --endpoint "${ENDPOINT}" \
  --run-dir "${RUN_DIR}" \
  --tier "${TIER}" \
  --negative-test || true

echo "::: stage 4: vLLM inference ::::"
FALLBACK_FLAG=""
if [[ "${ALLOW_VLLM_FALLBACK}" == "1" ]]; then
  FALLBACK_FLAG="--allow-fallback"
fi
"${PY}" vllm_run.py \
  --model-dir "${DL_DIR}" \
  --tensor-parallel-size "${TP}" \
  --dtype bfloat16 \
  --max-tokens 256 \
  --batch 8 \
  --run-dir "${RUN_DIR}" \
  --tier "${TIER}" \
  ${FALLBACK_FLAG} || echo "[bench_remote] vllm stage exit=$?"

if [[ "${KEEP_OBJECTS}" == "1" ]]; then
  echo "::: stage 5: cleanup SKIPPED (KEEP_OBJECTS=1) ::::"
  echo "  model stays at ${ENDPOINT}/${BUCKET}/${PREFIX_ROOT}/${COLDKEY8}/"
else
  echo "::: stage 5: cleanup tier prefix ::::"
  "${PY}" cleanup.py \
    --bucket "${BUCKET}" \
    --prefix "${PREFIX_ROOT}/${COLDKEY8}/" \
    --endpoint "${ENDPOINT}" || true
fi

echo "[bench_remote] tier=${TIER} done."
