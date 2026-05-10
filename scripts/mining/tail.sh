#!/usr/bin/env bash
# =============================================================================
# tail.sh — convenience helper for the tmux-managed training run.
#
# Subcommands:
#   ./tail.sh             — attach to the live tmux session (Ctrl-b d to detach)
#   ./tail.sh log         — `tail -f` the WORK_DIR/train.log (read-only stream)
#   ./tail.sh status      — quick snapshot: tmux session + last 40 log lines +
#                            current verdict.json (if any)
#   ./tail.sh stop        — kill the tmux session (training process terminates)
#   ./tail.sh verdict     — pretty-print WORK_DIR/verdict.json
# =============================================================================
set -euo pipefail

# shellcheck source=_lib.sh
source "$(dirname "$0")/_lib.sh"
cd "$_LIB_DIR"

_load_env >/dev/null

cmd="${1:-attach}"

case "$cmd" in
  attach|"" )
    tmux attach -t "$TMUX_SESSION" \
      || _die "no tmux session '$TMUX_SESSION'. Did you run ./start.sh?"
    ;;

  log )
    [ -f "$WORK_DIR/train.log" ] || _die "no log at $WORK_DIR/train.log"
    tail -f "$WORK_DIR/train.log"
    ;;

  status )
    echo "--- tmux ---"
    tmux ls 2>/dev/null | grep -E "^${TMUX_SESSION}\b" \
      || echo "(no '$TMUX_SESSION' session — training is not running)"
    echo
    echo "--- last 40 log lines ($WORK_DIR/train.log) ---"
    tail -n 40 "$WORK_DIR/train.log" 2>/dev/null || echo "(no log yet)"
    echo
    echo "--- verdict ($WORK_DIR/verdict.json) ---"
    if [ -f "$WORK_DIR/verdict.json" ]; then
      python -m json.tool < "$WORK_DIR/verdict.json"
    else
      echo "(no verdict yet)"
    fi
    ;;

  stop|kill )
    tmux kill-session -t "$TMUX_SESSION" 2>/dev/null \
      && _info "killed tmux session '$TMUX_SESSION'" \
      || _info "no tmux session '$TMUX_SESSION' to kill"
    ;;

  verdict )
    [ -f "$WORK_DIR/verdict.json" ] || _die "no verdict at $WORK_DIR/verdict.json"
    python -m json.tool < "$WORK_DIR/verdict.json"
    ;;

  * )
    cat <<EOF
usage: $0 [attach|log|status|stop|verdict]

  attach   attach to tmux session '$TMUX_SESSION' (Ctrl-b d to detach)
  log      tail -f \$WORK_DIR/train.log (no tmux interaction)
  status   one-shot summary: tmux + last 40 log lines + verdict
  stop     kill the tmux session (terminates training)
  verdict  pretty-print verdict.json
EOF
    exit 2
    ;;
esac
