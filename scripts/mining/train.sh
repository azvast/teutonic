#!/usr/bin/env bash
# =============================================================================
# train.sh — Manual real-training launcher (no tmux, no prompts, plain logs).
#
# Use this when you want to:
#   - Read the *actual* training command and tweak flags inline.
#   - Run from an existing tmux/screen pane you manage yourself.
#   - Debug a failed start.sh run with foreground output.
#
# This script does NOT manage tmux. RECOMMENDED — wrap in tmux yourself so
# SSH disconnects don't kill your run:
#   tmux new -s miner './train.sh'
#   # later, reattach with:
#   tmux attach -t miner
#
# To save logs as well:
#   tmux new -s miner './train.sh 2>&1 | tee /root/teutonic-mining/work/train.log'
#
# For a "set-and-forget" production run (tmux + log file + verdict checks),
# use ./start.sh instead.
#
# Usage:
#   ./train.sh [TAG]
#       TAG defaults to "v1". HF repo becomes:
#         $HF_ACCOUNT/Teutonic-LXXX-$COLDKEY_PREFIX-$TAG
# =============================================================================
set -euo pipefail

# shellcheck source=_lib.sh
source "$(dirname "$0")/_lib.sh"
cd "$_LIB_DIR"

_load_env
_activate_venv

_require_var HF_TOKEN
_require_var HF_ACCOUNT
_require_var COLDKEY_PREFIX
_require_var N_GPUS
_require_var WORK_DIR
_require_var BUNDLE_DIR
_require_cmd torchrun

TAG="${1:-v1}"
UPLOAD_REPO="$(_default_upload_repo "$TAG")"
WB_RUN_NAME="${WANDB_RUN_NAME:-real-${TAG}-$(date +%Y%m%d-%H%M)}"

case "$HF_TOKEN" in
  hf_REPLACE_ME|"") _die "HF_TOKEN looks like a placeholder. Edit .env." ;;
esac

if ! [[ "${UPLOAD_REPO,,}" == *"${COLDKEY_PREFIX,,}"* ]]; then
  _die "UPLOAD_REPO=$UPLOAD_REPO does not contain COLDKEY_PREFIX=$COLDKEY_PREFIX. Validator will reject."
fi

mkdir -p "$WORK_DIR"
unset EVAL_DELTA || true   # never let stale shell vars override .env

# ---------------------------------------------------------------------------
# THE ACTUAL COMMAND
# ---------------------------------------------------------------------------
# train_challenger.py is the orchestrator (single python process).
# It internally launches `torchrun --nproc_per_node=$N_GPUS` for the inner
# LoRA trainer. Do NOT wrap this command in torchrun.
#
# Tweak any of the flags below to experiment. Comments explain why each
# value was chosen for the live delta=0.0025 acceptance floor.
# ---------------------------------------------------------------------------
echo "[train] $(date '+%F %T') launching:"
echo "  upload-repo : $UPLOAD_REPO"
echo "  N_GPUS      : $N_GPUS"
echo "  WORK_DIR    : $WORK_DIR"
echo "  W&B run     : $WB_RUN_NAME (project=${WANDB_PROJECT:-<unset>})"
echo

exec python train_challenger.py \
  `# === I/O ============================================================` \
  --work          "$WORK_DIR"                                              \
  --bundle        "$BUNDLE_DIR"                                            \
  --report-out    "$WORK_DIR/verdict.json"                                 \
  --upload-repo   "$UPLOAD_REPO"                                           \
  --hf-token      "$HF_TOKEN"                                              \
  `# === topology: how many GPUs / iters / samples =====================` \
  --n-gpus           "$N_GPUS"     `# 4xB200 -> 4. Inner torchrun uses all.` \
  --n-shards         8             `# 8 token-shards loaded into memory`    \
  --shard-start      0                                                     \
  --eval-shard       10            `# held-out shard for paired eval`       \
  --n-score          8000          `# samples scored by king for curation`  \
  --train-per-iter   8000          `# training samples per iteration`       \
  --val-size         400           `# per-iter val (early-stop signal)`     \
  --n-eval           5000          `# paired-eval samples vs delta=0.0025`  \
  --max-iters        5             `# up to 5 iters; stops early if target hit` \
  --warm-start-iters               `# iter N starts from iter N-1's LoRA`   \
  --target-mu        0.012         `# ~5x delta — bootstrap can certify`    \
  --target-lcb       0.005         `# 2x delta — shard-variance buffer`     \
  `# === optimizer / scheduler =========================================` \
  --micro-batch      1                                                     \
  --grad-accum       16            `# effective batch = N_GPUS * mb * ga`   \
  --lr               1e-4                                                  \
  --epochs           1.5                                                   \
  --warmup-ratio     0.05                                                  \
  --weight-decay     0.01                                                  \
  --max-grad-norm    1.0                                                   \
  --adam-beta1       0.9                                                   \
  --adam-beta2       0.95                                                  \
  --lr-scheduler     cosine                                                \
  --optim            paged_adamw_8bit  `# saves ~10GiB vs adamw_torch_fused` \
  `# === LoRA ==========================================================` \
  --lora-r           32                                                    \
  --lora-alpha       64                                                    \
  --lora-dropout     0.0           `# MUST be 0; PEFT ParamWrapper rejects >0` \
  --lora-rslora                                                            \
  `# === inner-trainer log cadence =====================================` \
  --logging-steps    5                                                     \
  --eval-steps-inner 25                                                    \
  --save-steps-inner 100                                                   \
  `# === W&B ===========================================================` \
  --wandb-run-name "$WB_RUN_NAME"                                          \
  --wandb-tags     "real,$TAG,$(uname -n),${BT_WALLET_NAME:-nokey}-${BT_WALLET_HOTKEY:-nokey}" \
  ${WANDB_PROJECT:+--wandb-project "$WANDB_PROJECT"}                       \
  ${WANDB_ENTITY:+--wandb-entity  "$WANDB_ENTITY"}
