#!/usr/bin/env bash
# =============================================================================
# eval.sh — re-run the offline paired bootstrap test on an existing merged
# challenger directory against the CURRENT live king.
#
# Why this exists:
#   - The king on chain can change between the time start.sh produced its
#     verdict and the time you decide to push. If the king has changed,
#     your old verdict is stale — re-evaluate before burning a HF push +
#     on-chain TAO.
#   - Lets you test a merged candidate that came from a different training
#     run (e.g. you copy /root/teutonic-mining/work/iter_03/merged from
#     another run for re-evaluation).
#
# Reuses train_challenger.py's paired_eval() (same shard slicing + bootstrap
# math the validator uses) so results are apples-to-apples.
#
# Usage:
#   ./eval.sh <merged_dir> [n_eval] [eval_shard]
#
#   merged_dir   path to a candidate model (must contain config.json +
#                *.safetensors)
#   n_eval       number of paired sequences (default 5000)
#   eval_shard   held-out shard index (default 10, never overlap your
#                training shards)
#
# Output:
#   prints the JSON verdict to stdout
#   writes verdict to <merged_dir>/eval-verdict-<timestamp>.json
# =============================================================================
set -euo pipefail

# shellcheck source=_lib.sh
source "$(dirname "$0")/_lib.sh"
cd "$_LIB_DIR"

_load_env
_activate_venv

[ $# -lt 1 ] && _die "usage: $0 <merged_dir> [n_eval] [eval_shard]"

MERGED_DIR="$(realpath "$1")"
N_EVAL="${2:-5000}"
EVAL_SHARD="${3:-10}"

[ -d "$MERGED_DIR" ] || _die "merged_dir does not exist: $MERGED_DIR"
[ -f "$MERGED_DIR/config.json" ] || _die "no config.json in $MERGED_DIR"

_require_var WORK_DIR
mkdir -p "$WORK_DIR/cache"
unset EVAL_DELTA || true

OUT="$MERGED_DIR/eval-verdict-$(date +%Y%m%d-%H%M%S).json"

_info "running paired_eval against the LIVE king"
_info "  merged_dir : $MERGED_DIR"
_info "  n_eval     : $N_EVAL  (validator uses ~20000)"
_info "  eval_shard : $EVAL_SHARD"
_info "  output     : $OUT"

python - "$MERGED_DIR" "$N_EVAL" "$EVAL_SHARD" "$WORK_DIR" "$OUT" <<'PY'
import json, sys, os, shutil
from pathlib import Path
import numpy as np
from huggingface_hub import snapshot_download

# Borrow paired_eval + helpers from the harness (which auto-loads .env
# from this same directory and registers the active arch with HF Auto*).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "mining"))
from train_challenger import (  # noqa: E402
    fetch_king, fetch_manifest, download_shard, load_shard,
    paired_eval, sha256_dir, EVAL_DELTA,
)

merged_dir, n_eval, eval_shard, work_dir, out_path = sys.argv[1:]
n_eval = int(n_eval)
eval_shard = int(eval_shard)
work = Path(work_dir)
cache = work / "cache"
cache.mkdir(parents=True, exist_ok=True)

# 1. Pull the LIVE king (re-downloaded; might differ from the one used
# during training if a dethrone happened).
king = fetch_king()
king_dir = work / "king-eval"
if king_dir.exists():
    shutil.rmtree(king_dir)
print(f"[eval] downloading king -> {king_dir}")
snapshot_download(king["hf_repo"], local_dir=str(king_dir),
                  revision=king.get("king_revision") or None,
                  token=os.environ.get("HF_TOKEN") or None,
                  max_workers=16)
king_hash = sha256_dir(king_dir)
print(f"[eval] king sha256[:16]={king_hash[:16]}")

# 2. Eval shard (same selection the orchestrator uses by default).
manifest = fetch_manifest(cache)
key = manifest["shards"][eval_shard]["key"]
path = cache / Path(key).name
download_shard(key, path)
arr, _ = load_shard(path)
rng = np.random.default_rng(0xE1A)
indices = rng.choice(len(arr), size=min(n_eval, len(arr)), replace=False).tolist()
print(f"[eval] shard {eval_shard}: {len(arr)} sequences (sampling {len(indices)})")
print(f"[eval] EVAL_DELTA = {EVAL_DELTA}")

# 3. Paired bootstrap test.
verdict = paired_eval(str(king_dir), merged_dir, arr, indices, "cuda:0")
verdict["king_repo"] = king["hf_repo"]
verdict["king_revision"] = king.get("king_revision")
verdict["king_hash"] = king_hash
verdict["challenger_dir"] = merged_dir
verdict["challenger_hash"] = sha256_dir(Path(merged_dir))
verdict["n_eval"] = n_eval
verdict["eval_shard"] = eval_shard

with open(out_path, "w") as f:
    json.dump(verdict, f, indent=2, default=float)

print()
print(json.dumps(verdict, indent=2, default=float))
print()
print(f"[eval] wrote -> {out_path}")
PY
