#!/usr/bin/env bash
# =============================================================================
# noise.sh — minimum-viable end-to-end test of the validator pipeline.
#
# Uses miner.py to:
#   1. download the current king from HF
#   2. perturb every learnable tensor by Gaussian noise (default 1e-4)
#   3. push the noisy weights to your HF account
#   4. submit the reveal on-chain
#
# Then your hotkey shows up on https://teutonic.ai/dashboard.json with a
# verdict in ~10 minutes. You will almost CERTAINLY NOT dethrone — the
# point is to validate that:
#   - your HF token has write access
#   - the validator can reach + load your repo
#   - your coldkey-prefix repo naming is correct
#   - your chain reveal lands cleanly
#
# Costs: ~$0 of GPU (no training), ~165 GiB HF push, tiny TAO for the
# subnet commit.
#
# Usage:
#   ./noise.sh [NOISE]
#
#   NOISE — Gaussian stdev, default 1e-4. Don't go above ~1e-3 or the
#           trainability probe in eval/torch_runner.py rejects you with
#           "loss_non_finite:nan".
# =============================================================================
set -euo pipefail

# shellcheck source=_lib.sh
source "$(dirname "$0")/_lib.sh"
cd "$_LIB_DIR"

_load_env
_activate_venv

_require_var HF_TOKEN
_require_var BT_WALLET_NAME
_require_var BT_WALLET_HOTKEY
_require_var COLDKEY_PREFIX

NOISE="${1:-0.0001}"
SUFFIX="${COLDKEY_PREFIX}-noise-$(date +%H%M%S)"

# miner.py's repo target is "<seed_namespace>/<chain.NAME>-<suffix>".
# Show what that resolves to so you can spot misconfig early.
PREVIEW="$(cd "$_REPO_ROOT" && python -c "
import chain_config
ns = chain_config.SEED_NAMESPACE or 'unconst'
print(f'{ns}/{chain_config.NAME}-$SUFFIX')
")"

cat <<EOF

[mining] noise miner end-to-end test
  hotkey      : $BT_WALLET_NAME / $BT_WALLET_HOTKEY  (coldkey prefix: $COLDKEY_PREFIX)
  noise stdev : $NOISE
  HF target   : $PREVIEW   (note: lives under SEED_NAMESPACE; if you want
                            it under your own HF_ACCOUNT, edit miner.py
                            line 110-111 to use HF_ACCOUNT instead)
  on-chain    : YES (submits reveal commitment immediately)

EOF

read -r -p "[mining] proceed? (y/N) " ans
case "$ans" in
  y|Y|yes|YES) ;;
  *) _die "aborted." ;;
esac

cd "$_REPO_ROOT"
python miner.py \
  --hotkey "$BT_WALLET_HOTKEY" \
  --suffix "$SUFFIX" \
  --noise  "$NOISE"

_info "submitted. Watch https://teutonic.ai/dashboard.json — your hotkey should appear within ~30s."
_info "Possible verdicts:"
_info "  verdict.king         -> you didn't beat (expected for noise)"
_info "  verdict.error        -> coldkey gate failed, repo bad, or trainability NaN"
_info "  verdict.challenger   -> congrats, you somehow won with noise. Buy a lottery ticket."
