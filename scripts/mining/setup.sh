#!/usr/bin/env bash
# =============================================================================
# setup.sh — one-time bootstrap for a fresh GPU box.
#
# What it does:
#   1. Verifies CUDA / GPU / Python.
#   2. Creates and populates `.env` from `.env.example` if missing.
#   3. Creates a Python venv at <repo>/.venv and installs all deps.
#   4. Imports chain_config + the active arch to confirm transformers can
#      load Qwen3-MoE without trust_remote_code.
#   5. Creates the WORK_DIR and BUNDLE_DIR.
#   6. Reminds you of the manual steps the script can't do for you
#      (wallet creation, subnet registration, wandb/HF login).
#
# Usage:
#   bash setup.sh
#
# Run this ONCE per GPU box. Re-running is safe (idempotent).
# =============================================================================
set -euo pipefail

# shellcheck source=_lib.sh
source "$(dirname "$0")/_lib.sh"

cd "$_LIB_DIR"

# ---- 1. Bootstrap .env so subsequent steps can read it -------------------
if [ ! -f .env ]; then
  cp .env.example .env
  chmod 600 .env
  _info "copied .env.example -> .env (chmod 600)"
  _warn "EDIT scripts/mining/.env to set HF_TOKEN, COLDKEY_PREFIX, HF_ACCOUNT, then rerun ./setup.sh"
  exit 1
fi

# Source .env BEFORE we sanity-check, so WORK_DIR etc. are visible.
set -a
# shellcheck source=/dev/null
. .env
set +a

# ---- 2. Hardware sanity --------------------------------------------------
_require_cmd nvidia-smi
_info "CUDA devices:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || _die "nvidia-smi failed"

# ---- 3. Python detection -------------------------------------------------
PYTHON="$(command -v python3.12 || command -v python3.11 || command -v python3 || true)"
[ -n "$PYTHON" ] || _die "no python3.12 / python3.11 / python3 in PATH"
_info "python: $PYTHON ($($PYTHON --version))"

# ---- 4. venv -------------------------------------------------------------
if [ ! -d "$_REPO_ROOT/.venv" ]; then
  _info "creating venv at $_REPO_ROOT/.venv"
  "$PYTHON" -m venv "$_REPO_ROOT/.venv"
fi
# shellcheck source=/dev/null
. "$_REPO_ROOT/.venv/bin/activate"
pip install --upgrade pip wheel >/dev/null

# ---- 5. PyTorch (cu128 — required for B200/sm_100) ----------------------
_info "installing torch from cu128 wheel index..."
pip install --index-url https://download.pytorch.org/whl/cu128 torch

# ---- 6. Repo + mining-specific deps -------------------------------------
_info "installing teutonic repo (editable) + mining deps..."
pip install -e "$_REPO_ROOT"
pip install \
  peft \
  accelerate \
  bitsandbytes \
  wandb \
  hf_transfer \
  httpx

# ---- 7. Arch sanity (matches docs/MINING.md §1) -------------------------
_info "verifying chain_config + arch import..."
python - <<EOF
import sys, os
sys.path.insert(0, "$_REPO_ROOT")
import chain_config
print(f"[ok] chain.NAME={chain_config.NAME}  arch.MODULE={chain_config.ARCH_MODULE}  seed_repo={chain_config.SEED_REPO}")
chain_config.load_arch()
print("[ok] active arch registered with HF Auto*")
EOF

# ---- 8. Work dirs --------------------------------------------------------
mkdir -p "$WORK_DIR"
_info "WORK_DIR ready: $WORK_DIR (free: $(df -h "$WORK_DIR" | awk 'NR==2{print $4}'))"

# Mark sibling shell scripts executable (needed if you cloned on Windows
# and rsync'd to Linux without preserving +x).
for f in setup.sh smoke.sh start.sh eval.sh submit.sh noise.sh tail.sh; do
  [ -f "$_LIB_DIR/$f" ] && chmod +x "$_LIB_DIR/$f"
done
_info "made *.sh executable"

# Resolve BUNDLE_DIR (relative paths resolved from this script's dir).
case "$BUNDLE_DIR" in
  /*) bundle_abs="$BUNDLE_DIR" ;;
  *)  bundle_abs="$(cd "$_LIB_DIR/$BUNDLE_DIR" 2>/dev/null && pwd || echo "$BUNDLE_DIR")" ;;
esac
[ -d "$bundle_abs" ] || _warn "BUNDLE_DIR=$BUNDLE_DIR (resolved $bundle_abs) does not exist — train_lora_token_ids.py won't be found"

# ---- 9. Reminders --------------------------------------------------------
cat <<EOF

[mining] setup DONE.

Next steps (manual — not automated for safety):

  1. Sign in to W&B:
       wandb login

  2. Create your bittensor wallet:
       btcli wallet new_coldkey --wallet.name "$BT_WALLET_NAME"
       btcli wallet new_hotkey  --wallet.name "$BT_WALLET_NAME" --wallet.hotkey "$BT_WALLET_HOTKEY"
       btcli wallet list                     # note your coldkey ss58 -> first 8 chars

  3. Put the first 8 chars of your COLDKEY ss58 into scripts/mining/.env
     as COLDKEY_PREFIX (anti-impersonation gate; see docs/MINING.md §3).

  4. Register on SN3 (costs current registration burn):
       btcli subnet register --wallet.name "$BT_WALLET_NAME" \\
                             --wallet.hotkey "$BT_WALLET_HOTKEY" \\
                             --netuid "$TEUTONIC_NETUID" \\
                             --network "$TEUTONIC_NETWORK"

  5. Validate the pipeline (~20 min, ~\$10 of GPU):
       ./smoke.sh

  6. Real training (~hours, will push to HF if accepted):
       ./start.sh

EOF
