#!/usr/bin/env bash
# =============================================================================
# preflight.sh — local replay of the validator's reject-logic.
#
# Run BEFORE ./submit.sh (or against any HF repo you're curious about) to
# catch the most common rejection reasons miners hit:
#
#   - config_rejected   : arch/lock-key mismatch, auto_map, *.py shipped,
#                         non-canonical safetensors layout, oversized weights
#   - model_not_found   : repo unreachable, config.json missing, no .safetensors
#   - coldkey_required  : repo name doesn't embed your COLDKEY_PREFIX
#
# Reads HF_TOKEN and COLDKEY_PREFIX from .env automatically.
#
# Usage:
#   ./preflight.sh                                # uses $WORK_DIR/verdict.json
#   ./preflight.sh path/to/verdict.json
#   ./preflight.sh --hf-repo user/Teutonic-LXXX-... --revision <sha>
#   ./preflight.sh --hf-repo user/foo --strict    # exit nonzero on warnings too
#
# Exit code:
#   0 = would be accepted
#   1 = one or more reject-reasons found
# =============================================================================
set -euo pipefail

# shellcheck source=_lib.sh
source "$(dirname "$0")/_lib.sh"
cd "$_LIB_DIR"

_load_env
_activate_venv

# If first arg is a path to a verdict.json, forward as --verdict.
if [ $# -ge 1 ] && [ -f "$1" ]; then
  exec python "$_LIB_DIR/preflight.py" --verdict "$1" "${@:2}"
fi

# Otherwise: if --verdict / --hf-repo not in args, default to $WORK_DIR/verdict.json
if [[ "$*" != *"--verdict"* && "$*" != *"--hf-repo"* ]]; then
  if [ -n "${WORK_DIR:-}" ] && [ -f "$WORK_DIR/verdict.json" ]; then
    exec python "$_LIB_DIR/preflight.py" --verdict "$WORK_DIR/verdict.json" "$@"
  fi
  _die "no verdict.json found at \$WORK_DIR/verdict.json — pass a path or use --hf-repo"
fi

exec python "$_LIB_DIR/preflight.py" "$@"
