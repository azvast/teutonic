#!/usr/bin/env python3
"""Local preflight: same checks the validator runs before accepting a repo.

Mirrors `validator.validate_challenger_config` + the coldkey-prefix gate
from `validator.evaluate_one`. Run AFTER uploading to HF and BEFORE
submitting on-chain to catch the most common "submitted but rejected"
failure modes:

    - config_rejected (architecture mismatch, vocab/dim mismatch, lock-key
      mismatch, auto_map present, custom *.py shipped, oversized weights)
    - model_not_found (HF repo unreachable, config.json missing, no
      safetensors files, non-canonical safetensors layout)
    - coldkey_required (repo name doesn't embed your COLDKEY_PREFIX)

Returns a list of human-readable rejection reasons (empty list = OK).

Usage as a library:
    from preflight import preflight_check
    reasons = preflight_check(hf_repo="me/foo", revision="abc123",
                               king_repo="kabb/bar", king_revision="def456",
                               coldkey_prefix="5F1EGixU")
    if reasons:
        print("WOULD BE REJECTED:")
        for r in reasons: print(" -", r)

Usage as a CLI:
    python preflight.py \\
        --hf-repo me/Teutonic-LXXX-5F1EGixU-v1 \\
        --revision abc1234 \\
        --king-repo kabb/Teutonic-LXXX-...-king \\
        --king-revision def5678 \\
        --coldkey-prefix 5F1EGixU
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Iterable

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    import chain_config  # noqa: E402
    _CHAIN_NAME = chain_config.NAME
    _SEED_REPO = chain_config.SEED_REPO
    _EXTRA_LOCK_KEYS: tuple[str, ...] = tuple(chain_config.EXTRA_LOCK_KEYS)
except Exception as e:  # noqa: BLE001
    print(f"[preflight] WARNING: chain_config import failed ({e}); "
          "using generic lock keys only.", file=sys.stderr)
    _CHAIN_NAME = "unknown"
    _SEED_REPO = ""
    _EXTRA_LOCK_KEYS = ()


# Same canonical set the validator uses for sharded safetensors.
_SAFETENSORS_SHARD_RE = re.compile(r"^model-\d{5}-of-\d{5}\.safetensors$")

# Same total-size cap the validator uses (override with the same env var
# so both sides agree).
_MAX_SAFETENSORS_GB = float(os.environ.get(
    "TEUTONIC_MAX_CHALLENGER_SAFETENSORS_GB", "200"))

# Same generic structural lock the validator uses. Keep in lockstep with
# validator.py:_generic_lock — any miss here means a successful preflight
# can still fail at the validator.
_GENERIC_LOCK_KEYS: tuple[str, ...] = (
    "vocab_size", "hidden_size", "num_hidden_layers",
    "num_attention_heads", "num_key_value_heads", "head_dim",
    "intermediate_size", "model_type",
    "tie_word_embeddings", "rope_theta", "max_position_embeddings",
    "max_seq_len",
)


def _fetch_config(api, repo: str, revision: str, hf_token: str | None) -> dict | None:
    try:
        path = api.hf_hub_download(repo, "config.json",
                                    token=hf_token or None,
                                    revision=revision or None)
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def preflight_check(hf_repo: str,
                    revision: str = "",
                    king_repo: str = "",
                    king_revision: str = "",
                    coldkey_prefix: str = "",
                    hf_token: str | None = None) -> list[str]:
    """Returns a list of rejection reasons. Empty list == would-be-accepted.

    Every reason string here is shaped to match what `validator.py` would
    produce, so you can grep validator logs against them for parity.
    """
    reasons: list[str] = []

    try:
        from huggingface_hub import HfApi
    except ImportError:
        return ["preflight: huggingface_hub not installed"]

    # ---- 0. Coldkey-prefix gate ------------------------------------------
    if coldkey_prefix:
        if coldkey_prefix.lower() not in hf_repo.lower():
            reasons.append(
                f"hf repo '{hf_repo}' must contain miner coldkey prefix "
                f"'{coldkey_prefix}' (case-insensitive substring); "
                f"validator will reject as 'coldkey_required'.")

    api = HfApi(token=hf_token or os.environ.get("HF_TOKEN") or None)

    # ---- 1. Reachability + config.json ------------------------------------
    try:
        repo_files = api.list_repo_files(hf_repo, token=hf_token or None,
                                          revision=revision or None)
    except Exception as e:
        # Validator treats this as "config_rejected (cannot fetch config.json)"
        # because its only entry point is config.json; we surface it explicitly.
        reasons.append(f"repo unreachable ({type(e).__name__}: {e}); "
                       "validator will reject as 'cannot fetch config.json'")
        # No point continuing — every subsequent check needs the file listing.
        return reasons

    chall_cfg = _fetch_config(api, hf_repo, revision, hf_token)
    if chall_cfg is None:
        reasons.append("cannot fetch config.json (file missing or HF "
                       "permissions block read at this revision)")
        return reasons

    # ---- 2. Architecture / structural lock keys --------------------------
    king_cfg: dict = {}
    if king_repo:
        king_cfg = _fetch_config(api, king_repo, king_revision, hf_token) or {}
    if not king_cfg and _SEED_REPO:
        king_cfg = _fetch_config(api, _SEED_REPO, "", hf_token) or {}

    if king_cfg:
        king_arch = king_cfg.get("architectures", [])
        chall_arch = chall_cfg.get("architectures", [])
        if king_arch and chall_arch and king_arch != chall_arch:
            reasons.append(
                f"architecture mismatch: king={king_arch} challenger={chall_arch}")

        for key in _GENERIC_LOCK_KEYS + _EXTRA_LOCK_KEYS:
            king_val = king_cfg.get(key)
            chall_val = chall_cfg.get(key)
            if king_val is not None and chall_val is not None and king_val != chall_val:
                reasons.append(
                    f"{key} mismatch: king={king_val} challenger={chall_val}")
    else:
        reasons.append("preflight: could not fetch king config for "
                       "architecture/lock-key comparison — validator may "
                       "still reject if you've changed any of: "
                       + ", ".join(_GENERIC_LOCK_KEYS + _EXTRA_LOCK_KEYS))

    # ---- 3. Policy: no custom code ---------------------------------------
    if "auto_map" in chall_cfg:
        reasons.append("auto_map present in config.json "
                       "(custom modeling code is not allowed)")

    py_files = [f for f in repo_files if f.endswith(".py")]
    if py_files:
        reasons.append(f"repo ships *.py files (not allowed): {py_files[:3]}")

    # ---- 4. Safetensors layout -------------------------------------------
    st_files = [s for s in repo_files if s.endswith(".safetensors")]
    if not st_files:
        reasons.append("no .safetensors files in repo")
    else:
        has_single = "model.safetensors" in repo_files
        has_index = "model.safetensors.index.json" in repo_files
        has_shards = any(_SAFETENSORS_SHARD_RE.match(f) for f in st_files)
        if not (has_single or (has_index and has_shards)):
            if has_shards and not has_index:
                shard_count = sum(1 for f in st_files
                                  if _SAFETENSORS_SHARD_RE.match(f))
                reasons.append(
                    f"missing `model.safetensors.index.json` for sharded "
                    f"safetensors layout (found {shard_count} "
                    f"`model-NNNNN-of-NNNNN.safetensors` shards but no index "
                    f"file — `from_pretrained` cannot load shards without it)")
            else:
                reasons.append(
                    f"safetensors files present but none match the canonical "
                    f"transformers layout (expected `model.safetensors` OR "
                    f"`model.safetensors.index.json` + sharded "
                    f"`model-NNNNN-of-NNNNN.safetensors`); got {st_files[:3]}")

    # ---- 5. Total safetensors size cap -----------------------------------
    try:
        info = api.repo_info(hf_repo, token=hf_token or None,
                              revision=revision or None, files_metadata=True)
        total_st_bytes = sum((s.size or 0) for s in (info.siblings or [])
                              if s.rfilename.endswith(".safetensors"))
    except Exception:
        total_st_bytes = 0

    if total_st_bytes > 0:
        size_gb = total_st_bytes / 1e9
        if size_gb > _MAX_SAFETENSORS_GB:
            reasons.append(
                f"oversized: {size_gb:.1f} GB of .safetensors > "
                f"{_MAX_SAFETENSORS_GB:.0f} GB cap (normal Qwen3MoE 80B is "
                f"~154 GB; check for fp32 weights, duplicated shards, or "
                f"extra optimizer state)")

    return reasons


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _from_verdict(verdict_path: str) -> dict:
    with open(verdict_path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf-repo", default="",
        help="Challenger HF repo (e.g. me/Teutonic-LXXX-5F1EGixU-v1). "
             "Optional if --verdict is given.")
    ap.add_argument("--revision", default="",
        help="Optional commit SHA to pin (defaults to latest).")
    ap.add_argument("--king-repo", default="",
        help="King HF repo for arch/lock-key comparison. "
             "Optional if --verdict is given.")
    ap.add_argument("--king-revision", default="",
        help="Optional king commit SHA.")
    ap.add_argument("--coldkey-prefix", default=os.environ.get("COLDKEY_PREFIX", ""),
        help="First 8 ss58 chars of your coldkey. Reads $COLDKEY_PREFIX by default.")
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""),
        help="HF token. Reads $HF_TOKEN by default. Optional for public repos.")
    ap.add_argument("--verdict", default="",
        help="Read hf-repo / revision / king-repo / king-revision from a "
             "verdict.json produced by train_challenger.py.")
    ap.add_argument("--strict", action="store_true",
        help="Exit nonzero on ANY reason (default: only on hard rejections "
             "— preflight-only soft warnings exit 0).")
    args = ap.parse_args()

    hf_repo, revision = args.hf_repo, args.revision
    king_repo, king_revision = args.king_repo, args.king_revision
    if args.verdict:
        v = _from_verdict(args.verdict)
        hf_repo = hf_repo or v.get("uploaded_repo", "")
        revision = revision or v.get("uploaded_revision", "")
        king_repo = king_repo or v.get("king_repo", "")
        king_revision = king_revision or v.get("king_revision", "")

    if not hf_repo:
        ap.error("--hf-repo or --verdict required")

    print(f"[preflight] hf_repo       = {hf_repo}", file=sys.stderr)
    print(f"[preflight] revision      = {revision or '<latest>'}", file=sys.stderr)
    print(f"[preflight] king_repo     = {king_repo or '<seed/dashboard>'}", file=sys.stderr)
    print(f"[preflight] coldkey_prefix= {args.coldkey_prefix or '<unset — skipping>'}", file=sys.stderr)
    print(f"[preflight] chain         = {_CHAIN_NAME} (lock keys: {len(_EXTRA_LOCK_KEYS)} extra)", file=sys.stderr)

    reasons = preflight_check(
        hf_repo=hf_repo,
        revision=revision,
        king_repo=king_repo,
        king_revision=king_revision,
        coldkey_prefix=args.coldkey_prefix,
        hf_token=args.hf_token,
    )

    if not reasons:
        print("[preflight] ✓ all checks passed — validator should accept this repo.",
              file=sys.stderr)
        sys.exit(0)

    print(f"[preflight] ✗ {len(reasons)} reason(s) the validator would REJECT this repo:",
          file=sys.stderr)
    for r in reasons:
        print(f"  - {r}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
