# shellcheck shell=bash
# =============================================================================
# Shared helpers for the mining shell scripts.
#
# Sourced by setup.sh / smoke.sh / start.sh / eval.sh / submit.sh / noise.sh.
# Provides:
#   _load_env       — sources scripts/mining/.env (KEY=VALUE only)
#   _activate_venv  — activates ../.venv if present
#   _require_var V  — fails loudly if $V is unset or still a placeholder
#   _require_cmd C  — fails loudly if command C is missing
#   _info / _warn / _die — logging helpers
# =============================================================================

_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO_ROOT="$(cd "$_LIB_DIR/../.." && pwd)"

_info() { printf '\033[36m[mining]\033[0m %s\n' "$*"; }
_warn() { printf '\033[33m[mining]\033[0m WARNING: %s\n' "$*" >&2; }
_die()  { printf '\033[31m[mining]\033[0m FATAL: %s\n' "$*" >&2; exit 1; }

_load_env() {
  local f="$_LIB_DIR/.env"
  if [ ! -f "$f" ]; then
    _die "no .env at $f. Run ./setup.sh or 'cp .env.example .env' and edit it."
  fi
  # `set -a` auto-exports every var defined while sourcing.
  set -a
  # shellcheck source=/dev/null
  . "$f"
  set +a
  _info "loaded env from $f"
}

_activate_venv() {
  if [ -f "$_REPO_ROOT/.venv/bin/activate" ]; then
    # shellcheck source=/dev/null
    . "$_REPO_ROOT/.venv/bin/activate"
    _info "activated venv at $_REPO_ROOT/.venv"
  else
    _warn "no .venv at $_REPO_ROOT/.venv — running with system python"
  fi
}

_require_var() {
  local v="$1"
  local val="${!v:-}"
  if [ -z "$val" ] || [[ "$val" == *REPLACE_ME* ]] || [[ "$val" == *XXXXXXX* ]]; then
    _die "$v is unset or still a placeholder. Edit scripts/mining/.env."
  fi
}

_require_cmd() {
  command -v "$1" >/dev/null 2>&1 || _die "missing required command: $1"
}

# Build the default upload-repo "<HF_ACCOUNT>/<chain.name>-<COLDKEY_PREFIX>-<tag>"
# from .env vars + chain.toml. Reads chain name via python so chain.toml
# stays the single source of truth.
_default_upload_repo() {
  local tag="${1:-v1}"
  local chain
  chain="$(cd "$_REPO_ROOT" && python -c 'import chain_config; print(chain_config.NAME)' 2>/dev/null || echo "Teutonic-LXXX")"
  printf '%s/%s-%s-%s' "$HF_ACCOUNT" "$chain" "$COLDKEY_PREFIX" "$tag"
}
