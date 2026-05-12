#!/usr/bin/env bash
# =============================================================================
# start.sh — REAL fine-tuning run against the live Teutonic king.
#
# Strategy (matches the math for delta=0.0025):
#   - LoRA-target ALL Qwen3-MoE expert FFN modules (not just attention).
#     The previous default silently froze all 13,824 expert FFN modules;
#     train_lora_token_ids.py + train_challenger.py now pass the right
#     chain-aware target list. See LORA_TARGET_PRESETS in those files.
#   - Higher LoRA rank (r=32, alpha=64, rsLoRA) for real capacity.
#   - paged_adamw_8bit + bf16 + gradient checkpointing -> fits comfortably
#     on 4xB200 (179 GB each).
#   - Validator floor delta=0.0025 nats/token requires mu_hat >> delta to
#     clear LCB. Targets:
#         offline mu_hat >= 0.012   (~5x delta — enough headroom for the
#                                    bootstrap to certify with N=5000)
#         offline lcb    >= 0.005   (2x delta — buffer for shard variance
#                                    on the validator's randomized shard)
#   - 5 iterations max with warm-start so each iter compounds the prior
#     adapter instead of starting from scratch.
#   - 5000 paired-eval sequences (matches the user's delta-bookkeeping).
#
# Runs inside tmux so SSH disconnects don't kill the run. Logs go to:
#   $WORK_DIR/train.log               (stdout/stderr of the training)
#   $WORK_DIR/verdict.json            (final JSON verdict)
#   wandb.ai/<entity>/<project>       (real-time per-step + per-iter metrics)
#
# After the run completes:
#   - inspect verdict.json
#   - if best.accepted == true and HF push happened, run ./submit.sh to
#     post the on-chain reveal commitment.
#
# Usage:
#   ./start.sh [TAG]
#
#   TAG (optional, default "v1") — appended to the HF repo name. Examples:
#     ./start.sh                  -> <HF_ACCOUNT>/Teutonic-LXXX-<COLDKEY_PREFIX>-v1
#     ./start.sh r48              -> <HF_ACCOUNT>/Teutonic-LXXX-<COLDKEY_PREFIX>-r48
#
# To monitor:   ./tail.sh
# To stop:      tmux kill-session -t "$TMUX_SESSION"
# =============================================================================
set -euo pipefail

# shellcheck source=_lib.sh
source "$(dirname "$0")/_lib.sh"
cd "$_LIB_DIR"

_load_env
_activate_venv

# Required vars for training + HF push (.env or shell). BT_WALLET_NAME and
# BT_WALLET_HOTKEY are NOT required here: this script never signs a
# bittensor tx, and those vars are only used as W&B labels. Submission
# happens elsewhere (./submit.sh, typically on a separate host that holds
# the wallet).
_require_var HF_TOKEN
_require_var HF_ACCOUNT
_require_var COLDKEY_PREFIX
_require_var N_GPUS
_require_var WORK_DIR
_require_var BUNDLE_DIR
_require_cmd torchrun
_require_cmd tmux

# Defaults so the W&B tag and confirmation banner have something to show
# when the GPU box has no wallet at all (training-only setup).
: "${BT_WALLET_NAME:=unknown}"
: "${BT_WALLET_HOTKEY:=unknown}"

TAG="${1:-v1}"
UPLOAD_REPO="$(_default_upload_repo "$TAG")"
WB_RUN_NAME="${WANDB_RUN_NAME:-real-${TAG}-$(date +%Y%m%d-%H%M)}"

# Sanity: refuse to start if .env still has placeholder secrets.
case "$HF_TOKEN" in
  hf_REPLACE_ME|"") _die "HF_TOKEN looks like a placeholder. Edit scripts/mining/.env." ;;
esac

# Sanity: validator's coldkey gate. submit_challenger.py also checks this,
# but failing now saves you ~hours of training before discovering you can't
# submit the result.
if ! [[ "${UPLOAD_REPO,,}" == *"${COLDKEY_PREFIX,,}"* ]]; then
  _die "UPLOAD_REPO=$UPLOAD_REPO does not contain COLDKEY_PREFIX=$COLDKEY_PREFIX (case-insensitive). Validator will reject with 'coldkey_required'."
fi

mkdir -p "$WORK_DIR"
TRAIN_LOG="$WORK_DIR/train.log"
VERDICT="$WORK_DIR/verdict.json"

cat <<EOF

[mining] launching REAL training run
  chain      : $(python -c 'import chain_config; print(chain_config.NAME)')
  king       : (auto-discovered from dashboard)
  wallet     : $BT_WALLET_NAME / $BT_WALLET_HOTKEY  (coldkey prefix: $COLDKEY_PREFIX)
  upload-repo: $UPLOAD_REPO
  WORK_DIR   : $WORK_DIR
  N_GPUS     : $N_GPUS
  W&B run    : $WB_RUN_NAME (project=${WANDB_PROJECT:-<unset>})
  tmux sess  : $TMUX_SESSION
  log file   : $TRAIN_LOG
  verdict    : $VERDICT

EOF

read -r -p "[mining] proceed? (y/N) " ans
case "$ans" in
  y|Y|yes|YES) ;;
  *) _die "aborted." ;;
esac

# Belt-and-braces: never let a stale EVAL_DELTA leak in from a prior shell.
unset EVAL_DELTA || true

# Build the python command. Heredoc array is easier to maintain than a
# giant single-line string.
#
# IMPORTANT: train_challenger.py is the *orchestrator* — single Python
# process. It internally launches `torchrun --nproc_per_node=N_GPUS` for
# the inner LoRA trainer (train_lora_token_ids.py). Wrapping the
# orchestrator in torchrun would fork it into N copies, each downloading
# the king and racing on `cuda:0`. Use plain `python` here.
PY_CMD=( python train_challenger.py
  # --- I/O ---
  --work          "$WORK_DIR"
  --bundle        "$BUNDLE_DIR"
  --report-out    "$VERDICT"
  --upload-repo   "$UPLOAD_REPO"
  --hf-token      "$HF_TOKEN"
  # --- iteration topology ---
  --n-gpus           "$N_GPUS"
  --n-shards         8
  --shard-start      0
  --eval-shard       10
  # Uncomment to skip the ~155 GiB king re-download when restarting after a
  # crash. Auto-falls back to fresh download if the king flipped on chain.
  # --reuse-king
  --n-score          8000
  --train-per-iter   8000
  --val-size         400
  # Larger eval shrinks the bootstrap CI (SE ~ 1/sqrt(N)). At 5000 we had
  # SE(lcb) ≈ 5e-4; at 8000 → ≈ 4e-4. Helps avoid offline-accept /
  # validator-reject when margins are this thin.
  --n-eval           8000
  --max-iters        5
  --warm-start-iters
  # Targets tuned for the live king (which won by mu_hat=0.0031, lcb=0.0028).
  # We want comfortable headroom over delta=0.0025 so the validator's
  # randomized 20k-shard re-eval still clears:
  #   mu_hat >= 0.005  (~2x delta — 70% headroom for shard variance)
  #   lcb    >= 0.003  (delta + 0.0005 — small but realistic buffer)
  # If you find these too easy/hard, tune up/down by ~0.001 at a time.
  --target-mu        0.005
  --target-lcb       0.003
  # --- optimizer / scheduler ---
  # mb=2 + ga=8 keeps effective batch the same as mb=1 + ga=16 but halves
  # the number of forward/backward passes per optimizer step. On B200 we
  # have ~25 GiB headroom per rank at mb=1 so mb=2 fits with margin.
  --micro-batch      2
  --grad-accum       8
  # Conservative LR for fine-tuning a strong king. 1e-4 is healthy for
  # fresh LoRA but tends to overshoot when chasing sub-1% nat-improvements.
  # Sweet spot for "beat-by-a-hair" mining is 3e-5 .. 7e-5.
  --lr               5e-5
  --epochs           2.0

  --warmup-ratio     0.05
  --weight-decay     0.01
  --max-grad-norm    1.0
  --adam-beta1       0.9
  --adam-beta2       0.95
  --lr-scheduler     cosine
  --optim            paged_adamw_8bit
  # --- LoRA ---
  # r=64 gives the LoRA enough capacity to represent useful expert-FFN
  # corrections in Qwen3-MoE (it has 128 experts × 3 projs × 64 layers).
  # rslora keeps the effective alpha bounded as r grows; without it the
  # gradient scale would explode at high ranks. Alpha = 2*r is the
  # rslora-compatible sweet spot.
  --lora-r           64
  --lora-alpha       128
  # PEFT requires lora_dropout == 0 when targets are nn.Parameter blocks
  # (Qwen3-MoE fused experts go through ParamWrapper). Do NOT raise this.
  --lora-dropout     0.0
  --lora-rslora
  # --- inner-trainer logging cadence ---
  --logging-steps    5
  --eval-steps-inner 25
  --save-steps-inner 100
  # --- W&B ---
  --wandb-run-name "$WB_RUN_NAME"
  --wandb-tags     "real,$TAG,$(uname -n),${BT_WALLET_NAME}-${BT_WALLET_HOTKEY}"
)
[ -n "${WANDB_PROJECT:-}" ] && PY_CMD+=( --wandb-project "$WANDB_PROJECT" )
[ -n "${WANDB_ENTITY:-}"  ] && PY_CMD+=( --wandb-entity  "$WANDB_ENTITY"  )

# Quote every arg so spaces inside paths survive the tmux send-keys path.
QUOTED=""
for a in "${PY_CMD[@]}"; do QUOTED+=" $(printf '%q' "$a")"; done

# Kill any stale session, then launch fresh inside tmux. Inside the tmux
# pane we re-source .env and re-activate the venv (the tmux child shell
# doesn't inherit `set -a` semantics from this script).
tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true
tmux new-session -d -s "$TMUX_SESSION" -c "$_LIB_DIR" \
  "set -a; . ./.env; set +a; \
   . '$_REPO_ROOT/.venv/bin/activate'; \
   $QUOTED 2>&1 | tee '$TRAIN_LOG'; \
   echo; echo '[mining] training process exited; pane left open for inspection'; \
   exec bash"

_info "tmux session '$TMUX_SESSION' started."
_info "follow logs: ./tail.sh   (or:  tmux attach -t $TMUX_SESSION)"
_info "verdict file (when done): $VERDICT"
_info "next: when verdict.best.accepted == true, run ./submit.sh"
