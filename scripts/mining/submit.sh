#!/usr/bin/env bash
# =============================================================================
# submit.sh — post the on-chain reveal commitment for an accepted challenger.
#
# Wraps submit_challenger.py. Reads verdict.json (produced by start.sh /
# train_challenger.py), validates that:
#   - best.accepted == true
#   - uploaded_repo + challenger_hash are present (HF push happened)
#   - HF repo embeds your COLDKEY_PREFIX (anti-impersonation gate)
# then submits the reveal payload `<king_hash[:16]>:<repo>:<chall_hash>`
# on subnet 3.
#
# Costs a small amount of TAO (subnet commit fee). Submission is final;
# the validator picks it up after the reveal block lands (~30s).
#
# Usage:
#   ./submit.sh                           # uses $WORK_DIR/verdict.json
#   ./submit.sh path/to/verdict.json
#   ./submit.sh path/to/verdict.json --dry-run   # validate, don't broadcast
# =============================================================================
set -euo pipefail

# shellcheck source=_lib.sh
source "$(dirname "$0")/_lib.sh"
cd "$_LIB_DIR"

_load_env
_activate_venv

_require_var BT_WALLET_NAME
_require_var BT_WALLET_HOTKEY
_require_var TEUTONIC_NETUID
_require_var TEUTONIC_NETWORK
_require_var WORK_DIR

VERDICT="${1:-$WORK_DIR/verdict.json}"
shift || true   # remaining args are forwarded to submit_challenger.py (e.g. --dry-run)

[ -f "$VERDICT" ] || _die "no verdict at: $VERDICT"

_info "verdict file: $VERDICT"
_info "wallet     : $BT_WALLET_NAME / $BT_WALLET_HOTKEY"
_info "subnet     : $TEUTONIC_NETUID  ($TEUTONIC_NETWORK)"

# Quick local sanity-print (submit_challenger.py also re-validates).
python - "$VERDICT" <<'PY'
import json, sys
v = json.load(open(sys.argv[1]))
b = v.get("best") or {}
print(f"  king_repo     : {v.get('king_repo')}")
print(f"  king_hash[:16]: {v.get('king_hash','')[:16]}")
print(f"  uploaded_repo : {v.get('uploaded_repo','<NOT UPLOADED>')}")
print(f"  uploaded_rev  : {v.get('uploaded_revision','<n/a>')[:12]}")
print(f"  chall_hash[:16]: {v.get('challenger_hash','')[:16]}")
print(f"  best.mu_hat   : {b.get('mu_hat')}")
print(f"  best.lcb      : {b.get('lcb')}  (delta={b.get('delta')})")
print(f"  best.accepted : {b.get('accepted')}")
if not b.get("accepted"):
    sys.exit("[error] best.accepted == False; refusing to submit (would burn TAO).")
if not v.get("uploaded_repo") or not v.get("challenger_hash"):
    sys.exit("[error] verdict missing uploaded_repo / challenger_hash; was --upload-repo set?")
PY

read -r -p "[mining] proceed with on-chain reveal? (y/N) " ans
case "$ans" in
  y|Y|yes|YES) ;;
  *) _die "aborted." ;;
esac

# submit_challenger.py owns the actual coldkey-prefix check + bittensor tx.
python "$_LIB_DIR/submit_challenger.py" \
  --verdict     "$VERDICT" \
  --wallet-name "$BT_WALLET_NAME" \
  --hotkey      "$BT_WALLET_HOTKEY" \
  --netuid      "$TEUTONIC_NETUID" \
  --network     "$TEUTONIC_NETWORK" \
  "$@"

_info "reveal submitted. Watch https://teutonic.ai/dashboard.json for your verdict."
