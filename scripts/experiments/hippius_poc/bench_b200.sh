#!/usr/bin/env bash
# Local driver: orchestrate the Hippius B200 perf benchmark.
#
# Usage:
#   ./bench_b200.sh                       # both tiers (tinyllama + quasar)
#   ./bench_b200.sh tinyllama             # smoke only
#   ./bench_b200.sh quasar                # headline only
#
# Required: Doppler authenticated for project=arbos. SSH key able to reach
# B200 box at the configured host:port (env-overridable).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
POC_DIR="${REPO_ROOT}/scripts/experiments/hippius_poc"

# ----- ssh config (lium box noble-hawk-85) -----
B200_USER="${B200_USER:-root}"
B200_HOST="${B200_HOST:-95.133.252.113}"
B200_PORT="${B200_PORT:-10300}"
B200_SSH_OPTS="${B200_SSH_OPTS:-"-o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ServerAliveCountMax=6"}"

POC_REMOTE="${POC_REMOTE:-/workspace/hippius_poc}"
CACHE_REMOTE="${CACHE_REMOTE:-/workspace}"

# ----- bench config -----
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$(openssl rand -hex 4)}"
PREFIX_ROOT="${PREFIX_ROOT:-poc/${RUN_ID}/challengers}"

# Backend: hippius | r2  (controls which doppler creds + endpoint + bucket).
BACKEND="${BACKEND:-hippius}"
case "${BACKEND}" in
  hippius)
    DEFAULT_BUCKET="teutonic-sn3"
    DEFAULT_ENDPOINT="https://s3.hippius.com"
    ;;
  r2)
    DEFAULT_BUCKET="$(doppler secrets get R2_NATIVE_BUCKET --plain -p arbos -c dev 2>/dev/null \
                     || doppler secrets get R2_BUCKET_NAME --plain -p arbos -c dev)"
    # Prefer the native R2 S3 endpoint; fall back to deriving from R2_BUCKET_URL.
    if R2_ENDPOINT_VAL="$(doppler secrets get R2_NATIVE_ENDPOINT --plain -p arbos -c dev 2>/dev/null)"; then
      DEFAULT_ENDPOINT="${R2_ENDPOINT_VAL}"
    else
      R2_BUCKET_URL_VAL="$(doppler secrets get R2_BUCKET_URL --plain -p arbos -c dev)"
      DEFAULT_ENDPOINT="${R2_BUCKET_URL_VAL%/*}"
    fi
    ;;
  *) echo "unknown BACKEND=${BACKEND} (use hippius or r2)" >&2; exit 1;;
esac
BUCKET="${BUCKET:-${DEFAULT_BUCKET}}"
ENDPOINT="${ENDPOINT:-${DEFAULT_ENDPOINT}}"
COLDKEY8="${COLDKEY8:-deadbeef}"

# KEEP_OBJECTS=1 skips the per-tier cleanup AND the final purge,
# so the uploaded model stays on Hippius for others to try.
KEEP_OBJECTS="${KEEP_OBJECTS:-0}"

# Per-tier overrides. Box has GPUs 0-3 in use by another process; default to
# the higher-indexed cards for safety. TP=1 fits both tiers comfortably on one B200.
TIER_TINYLLAMA_REPO="${TIER_TINYLLAMA_REPO:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
TIER_TINYLLAMA_TP="${TIER_TINYLLAMA_TP:-1}"
TIER_TINYLLAMA_GPUS="${TIER_TINYLLAMA_GPUS:-7}"
TIER_TINYLLAMA_GPU_UTIL="${TIER_TINYLLAMA_GPU_UTIL:-0.30}"

TIER_QUASAR_REPO="${TIER_QUASAR_REPO:-unconst/Teutonic-XXIV}"
TIER_QUASAR_TP="${TIER_QUASAR_TP:-1}"
TIER_QUASAR_GPUS="${TIER_QUASAR_GPUS:-6}"
TIER_QUASAR_GPU_UTIL="${TIER_QUASAR_GPU_UTIL:-0.50}"

LOCAL_RUN_DIR="${POC_DIR}/runs/${RUN_ID}"
mkdir -p "${LOCAL_RUN_DIR}"

START_TS=$(date +%s)
COST_PER_HR="${COST_PER_HR:-31.92}"

ssh_box() {
  ssh -p "${B200_PORT}" ${B200_SSH_OPTS} "${B200_USER}@${B200_HOST}" "$@"
}

scp_to_box() {
  scp -P "${B200_PORT}" ${B200_SSH_OPTS} -r "$1" "${B200_USER}@${B200_HOST}:$2"
}

scp_from_box() {
  scp -P "${B200_PORT}" ${B200_SSH_OPTS} -r "${B200_USER}@${B200_HOST}:$1" "$2"
}

# rsync_to_box: copy CONTENTS of src/ into dst/. Uses rsync if available,
# otherwise falls back to tar over ssh.
rsync_to_box() {
  local src="${1%/}/"
  local dst="${2%/}/"
  if ssh_box "command -v rsync >/dev/null 2>&1"; then
    rsync -az --delete -e "ssh -p ${B200_PORT} ${B200_SSH_OPTS}" "${src}" "${B200_USER}@${B200_HOST}:${dst}"
  else
    ssh_box "mkdir -p '${dst}'"
    tar -C "${src}" -cf - . \
      | ssh -p "${B200_PORT}" ${B200_SSH_OPTS} "${B200_USER}@${B200_HOST}" "tar -C '${dst}' -xf -"
  fi
}

purge_remote_prefix() {
  ssh_box "cd ${POC_REMOTE} 2>/dev/null && [ -f .venv/bin/activate ] && source .venv/bin/activate && \
    [ -f remote.env ] && source remote.env; \
    python cleanup.py --bucket '${BUCKET}' --prefix 'poc/${RUN_ID}/' --endpoint '${ENDPOINT}' 2>&1 || true; \
    python cleanup.py --purge-stale --max-age-hours 24 --endpoint '${ENDPOINT}' 2>&1 || true" || true
}

cleanup_on_exit() {
  local rc=$?
  local end_ts now_h cost
  end_ts=$(date +%s)
  now_h=$(awk -v a="${START_TS}" -v b="${end_ts}" 'BEGIN{printf "%.3f",(b-a)/3600.0}')
  cost=$(awk -v h="${now_h}" -v r="${COST_PER_HR}" 'BEGIN{printf "%.2f",h*r}')
  echo
  echo "[bench_b200] elapsed=${now_h}h estimated_cost=\$${cost} (rate \$${COST_PER_HR}/hr)"
  if [[ ${rc} -ne 0 && "${KEEP_OBJECTS}" != "1" ]]; then
    echo "[bench_b200] non-zero exit (${rc}); attempting Hippius prefix cleanup..."
    purge_remote_prefix || true
  elif [[ "${KEEP_OBJECTS}" == "1" ]]; then
    echo "[bench_b200] KEEP_OBJECTS=1: leaving poc/${RUN_ID}/ on Hippius for reuse."
  fi
  exit ${rc}
}
trap cleanup_on_exit EXIT INT TERM

# ----- doppler creds (run *locally* once and pass via SendEnv-style --env file) -----
echo "[bench_b200] resolving secrets from doppler (project=arbos, backend=${BACKEND})..."
HF_TOKEN_VAL="$(doppler secrets get HF_TOKEN --plain -p arbos -c prd)"

case "${BACKEND}" in
  hippius)
    AK_VAL="$(doppler secrets get HIPPIUS_ACCESS_KEY_ID --plain -p arbos -c dev 2>/dev/null \
              || doppler secrets get HIPPIUS_ACCESS_KEY_ID --plain -p arbos -c prd)"
    SK_VAL="$(doppler secrets get HIPPIUS_SECRET_ACCESS_KEY --plain -p arbos -c dev 2>/dev/null \
              || doppler secrets get HIPPIUS_SECRET_ACCESS_KEY --plain -p arbos -c prd)"
    ;;
  r2)
    AK_VAL="$(doppler secrets get R2_NATIVE_ACCESS_KEY_ID --plain -p arbos -c dev 2>/dev/null \
              || doppler secrets get R2_ACCESS_KEY_ID --plain -p arbos -c dev)"
    SK_VAL="$(doppler secrets get R2_NATIVE_SECRET_ACCESS_KEY --plain -p arbos -c dev 2>/dev/null \
              || doppler secrets get R2_SECRET_ACCESS_KEY --plain -p arbos -c dev)"
    if [[ "${#AK_VAL}" -lt 30 ]]; then
      echo "ERROR: R2 access key looks like a Hippius key (len=${#AK_VAL}, expected 32)." >&2
      echo "       Set R2_NATIVE_ACCESS_KEY_ID / R2_NATIVE_SECRET_ACCESS_KEY in doppler (project=arbos, config=dev)." >&2
      exit 1
    fi
    ;;
esac

if [[ -z "${AK_VAL}" || -z "${SK_VAL}" ]]; then
  echo "ERROR: missing ${BACKEND} credentials in doppler" >&2
  exit 1
fi

# ----- ensure remote dirs and source -----
echo "[bench_b200] preparing remote box ${B200_USER}@${B200_HOST}:${B200_PORT}..."
ssh_box "mkdir -p ${POC_REMOTE} ${CACHE_REMOTE}/hf_cache ${CACHE_REMOTE}/hippius_cache"
echo "[bench_b200] ensuring rsync on box (apt)..."
ssh_box "command -v rsync >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq rsync)" || echo "[warn] rsync install skipped"

echo "[bench_b200] rsync experiment dir + vendored arch..."
rsync_to_box "${POC_DIR}/" "${POC_REMOTE}/"
rsync_to_box "${REPO_ROOT}/archs/" "${POC_REMOTE}/archs/"

# ----- remote venv & deps -----
echo "[bench_b200] ensuring uv venv + deps on box..."
ssh_box "set -e; cd ${POC_REMOTE} && \
  if ! command -v uv >/dev/null 2>&1 && [ ! -x \$HOME/.local/bin/uv ]; then \
    curl -LsSf https://astral.sh/uv/install.sh | sh; \
  fi; \
  export PATH=\$HOME/.local/bin:\$PATH; \
  if [ ! -d .venv ]; then uv venv --python 3.12 .venv; fi; \
  source .venv/bin/activate && \
  uv pip install --quiet 'boto3>=1.34' 'botocore>=1.34' 'huggingface_hub>=0.24' \
    'hf_transfer>=0.1' 'safetensors>=0.4' 'transformers>=4.44' 'accelerate>=0.30' 'numpy<3' && \
  if ! python -c 'import vllm' 2>/dev/null; then \
    uv pip install --quiet 'vllm>=0.6.0' || echo '[warn] vllm install failed; will use transformers fallback'; \
  fi"

# ----- write remote env file -----
ENV_FILE_LOCAL="${LOCAL_RUN_DIR}/remote.env"
cat >"${ENV_FILE_LOCAL}" <<EOF
export HF_TOKEN='${HF_TOKEN_VAL}'
export HF_HUB_ENABLE_HF_TRANSFER='1'
export HF_HUB_DOWNLOAD_WORKERS='16'
export TEUTONIC_DS_ACCESS_KEY='${AK_VAL}'
export TEUTONIC_DS_SECRET_KEY='${SK_VAL}'
export TEUTONIC_DS_ENDPOINT='${ENDPOINT}'
export TEUTONIC_DS_BUCKET='${BUCKET}'
export BACKEND='${BACKEND}'
export RUN_ID='${RUN_ID}'
export PREFIX_ROOT='${PREFIX_ROOT}'
export BUCKET='${BUCKET}'
export ENDPOINT='${ENDPOINT}'
export COLDKEY8='${COLDKEY8}'
export POC_ROOT='${POC_REMOTE}'
export CACHE_ROOT='${CACHE_REMOTE}'
EOF
chmod 600 "${ENV_FILE_LOCAL}"
scp_to_box "${ENV_FILE_LOCAL}" "${POC_REMOTE}/remote.env"
ssh_box "chmod 600 ${POC_REMOTE}/remote.env"

# ----- run tiers -----
TIERS_TO_RUN=("${@:-tinyllama quasar}")
if [[ "${#TIERS_TO_RUN[@]}" -eq 1 && "${TIERS_TO_RUN[0]}" == "" ]]; then
  TIERS_TO_RUN=(tinyllama quasar)
fi

run_tier() {
  local tier="$1"
  local repo tp gpus util
  case "${tier}" in
    tinyllama)
      repo="${TIER_TINYLLAMA_REPO}"; tp="${TIER_TINYLLAMA_TP}"
      gpus="${TIER_TINYLLAMA_GPUS}"; util="${TIER_TINYLLAMA_GPU_UTIL}";;
    quasar)
      repo="${TIER_QUASAR_REPO}"; tp="${TIER_QUASAR_TP}"
      gpus="${TIER_QUASAR_GPUS}"; util="${TIER_QUASAR_GPU_UTIL}";;
    *) echo "unknown tier: ${tier}" >&2; return 2;;
  esac
  echo
  echo "================================================================="
  echo "[bench_b200] tier=${tier} repo=${repo} TP=${tp} GPUs=${gpus} util=${util} run_id=${RUN_ID}"
  echo "================================================================="
  ssh_box "set -e; cd ${POC_REMOTE} && source .venv/bin/activate && source ${POC_REMOTE}/remote.env && \
    export TIER='${tier}' HF_REPO='${repo}' TP='${tp}' \
           CUDA_VISIBLE_DEVICES='${gpus}' VLLM_GPU_UTIL='${util}' \
           KEEP_OBJECTS='${KEEP_OBJECTS}' && \
    bash bench_remote.sh"
}

for tier in "${TIERS_TO_RUN[@]}"; do
  run_tier "${tier}"
done

# ----- pull results -----
echo
echo "[bench_b200] pulling results..."
mkdir -p "${LOCAL_RUN_DIR}"
scp_from_box "${POC_REMOTE}/runs/${RUN_ID}/." "${LOCAL_RUN_DIR}/" || true
ls -la "${LOCAL_RUN_DIR}" || true

# ----- analyze -----
if command -v python3 >/dev/null; then
  python3 "${POC_DIR}/analyze.py" --run-dir "${LOCAL_RUN_DIR}" \
    | tee "${LOCAL_RUN_DIR}/RESULTS.md" || true
fi

echo "[bench_b200] run_id=${RUN_ID} local_run_dir=${LOCAL_RUN_DIR}"

# ----- final cleanup of poc/ prefix (skipped when KEEP_OBJECTS=1) -----
if [[ "${KEEP_OBJECTS}" == "1" ]]; then
  echo
  echo "[bench_b200] KEEP_OBJECTS=1 -- keeping poc/${RUN_ID}/ on Hippius."
  echo "[bench_b200] Share these URLs:"
  for f in "${LOCAL_RUN_DIR}"/commitment.*.json; do
    [[ -f "${f}" ]] || continue
    tier_name="$(basename "${f}" | sed 's/^commitment\.\(.*\)\.json$/\1/')"
    key=$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['s3_key'])" "${f}")
    echo "  ${tier_name}: ${ENDPOINT}/${BUCKET}/${PREFIX_ROOT}/${key}/"
  done
else
  purge_remote_prefix
fi
