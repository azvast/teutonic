#!/usr/bin/env bash
# =============================================================================
# smoke.sh — fast pipeline validation (~20-30 min, ~$10-15 of GPU).
#
# Tiny everything:
#   - 1 dataset shard (vs 8 in start.sh)
#   - 256 scoring + 256 training samples (vs ~8000 each)
#   - 0.05 epochs (just a few SGD steps)
#   - 256 eval sequences (vs 5000)
#   - 1 iteration (no warm-start retries)
#   - target_mu = 0.0 / target_lcb = -1.0  (so any verdict counts as "done")
#
# Goal: prove the WHOLE pipeline runs end-to-end on this box BEFORE you burn
# real GPU-hours:
#   king load -> shard download -> king-score curate -> torchrun-LoRA ->
#   merge -> paired_eval -> verdict.json + W&B run.
#
# Does NOT push to HF. Does NOT submit on-chain. Safe to run repeatedly.
#
# Usage:
#   ./smoke.sh                       # plain smoke run
#   ./smoke.sh --reuse-king          # skip re-downloading the king (~155 GiB)
#                                     # if it's already cached in $SMOKE_WORK/king
#   ./smoke.sh --max-iters 2 ...     # any other train_challenger.py flag is
#                                     # forwarded as-is (overrides smoke defaults)
#
# Note: smoke.sh uses its own work dir (`${WORK_DIR}-smoke`), so the first run
# downloads its own king. Subsequent `./smoke.sh --reuse-king` runs reuse it.
# To share start.sh's king cache, symlink it once:
#   ln -s /root/teutonic-mining/work/king /root/teutonic-mining/work-smoke/king
# =============================================================================
set -euo pipefail

# shellcheck source=_lib.sh
source "$(dirname "$0")/_lib.sh"
cd "$_LIB_DIR"

_load_env
_activate_venv

_require_var HF_TOKEN
_require_var N_GPUS
_require_var WORK_DIR
_require_var BUNDLE_DIR

# Smoke uses a separate work dir so it can't collide with start.sh state.
SMOKE_WORK="${WORK_DIR%/}-smoke"
mkdir -p "$SMOKE_WORK"

# W&B run name is fixed so repeated smoke runs append to a single dashboard.
WB_RUN_NAME="${WANDB_RUN_NAME:-smoke-$(date +%Y%m%d-%H%M%S)}"

_info "smoke run starting"
_info "  WORK_DIR    = $SMOKE_WORK"
_info "  BUNDLE_DIR  = $BUNDLE_DIR"
_info "  N_GPUS      = $N_GPUS"
_info "  WANDB run   = $WB_RUN_NAME (project=${WANDB_PROJECT:-<unset>})"

# Tip: unset EVAL_DELTA so the validator's default 0.0025 floor is used.
unset EVAL_DELTA || true

# IMPORTANT: train_challenger.py is the *orchestrator* — single Python
# process. It internally launches `torchrun --nproc_per_node=N_GPUS` for
# the inner LoRA trainer. Do NOT wrap the orchestrator in torchrun.
#
# Any extra CLI flags passed to smoke.sh (e.g. --reuse-king) are appended
# via "$@" as the LAST positional args. argparse honors the last
# occurrence of a repeated flag, so user overrides win.
# shellcheck disable=SC2086
python train_challenger.py \
  --work       "$SMOKE_WORK" \
  --bundle     "$BUNDLE_DIR" \
  --n-gpus     "$N_GPUS" \
  --n-shards         1 \
  --shard-start      0 \
  --eval-shard       10 \
  --n-score          256 \
  --train-per-iter   256 \
  --val-size         32 \
  --n-eval           256 \
  --max-iters        1 \
  --target-mu        0.0 \
  --target-lcb       -1.0 \
  --micro-batch      1 \
  --grad-accum       4 \
  --lr               1e-4 \
  --epochs           0.05 \
  --warmup-ratio     0.05 \
  --weight-decay     0.01 \
  --max-grad-norm    1.0 \
  --adam-beta2       0.95 \
  --lr-scheduler     cosine \
  --optim            adamw_torch_fused \
  --lora-r           8 \
  --lora-alpha       16 \
  --logging-steps    1 \
  --eval-steps-inner 5 \
  --save-steps-inner 50 \
  --report-out       "$SMOKE_WORK/verdict.json" \
  ${WANDB_PROJECT:+--wandb-project "$WANDB_PROJECT"} \
  ${WANDB_ENTITY:+--wandb-entity  "$WANDB_ENTITY"} \
  --wandb-run-name "$WB_RUN_NAME" \
  --wandb-tags     "smoke,${BT_WALLET_NAME:-nokey},$(uname -n)" \
  "$@"

echo
_info "smoke completed. Verdict:"
python -m json.tool < "$SMOKE_WORK/verdict.json" || cat "$SMOKE_WORK/verdict.json"

cat <<EOF

[mining] If you got this far without errors, the full pipeline works.
[mining] Next: ./start.sh   (real run; takes hours; pushes to HF if accepted)

EOF
