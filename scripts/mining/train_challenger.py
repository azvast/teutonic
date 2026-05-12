#!/usr/bin/env python3
"""Teutonic mining harness — train a challenger that beats the current king.

Runs on a multi-B200 box. Pipeline:
  1. Discover current king from R2 dashboard (repo + revision).
  2. Download king from HF (snapshot_download, pinned revision).
  3. Pull a few dataset shards from Hippius (pretokenized .npy uint32, seq_len=2048).
  4. Score sample sequences with the king (avg next-token loss).
  5. Build a curriculum (general / hard / easy buckets, drop suspicious).
  6. Train a LoRA adapter with torchrun multi-GPU on the chosen training mix.
  7. Merge LoRA into the base weights -> standalone candidate dir.
  8. Offline paired eval candidate vs king on a held-out shard slice
     (mirrors validator's compute_paired_losses + bootstrap LCB > delta).
  9. Emit a JSON verdict file. If accepted, optionally upload to HF.

Designed to be re-run iteratively (--max-iters): if first attempt's mu_hat
falls short, training is re-run with a different seed / more steps until
the budget is spent.

This script is meant to live on the GPU box (e.g. /root/teutonic-mining/)
and be invoked there. It does NOT touch bittensor — that step is handled
by submit_challenger.py on the templar host where the wallet lives.

REQUIRED — coldkey prefix in --upload-repo (since 2026-04-29):
  The validator rejects any HF repo whose name doesn't contain the first
  8 ss58 chars of the miner's coldkey (case-insensitive substring, in
  either the HF account or the model basename). This is an
  anti-impersonation gate — only the legit coldkey owner can publish a
  repo whose name embeds *their* coldkey. Imposters who lift somebody
  else's URL end up advertising the victim's coldkey on chain.

  This script doesn't have wallet access so it can't enforce locally —
  the orchestrator (run_pipeline.sh) and the on-chain submitter
  (submit_challenger.py) both check before they burn HF / TAO. Pass an
  --upload-repo matching the active chain (chain.toml -> [chain].name)
  with your coldkey prefix embedded, e.g.
      myaccount/<chain.name>-5DhAqMpd-v3
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import io
import json
import logging
import math
import os


# ---------------------------------------------------------------------------
# .env auto-loader — must run BEFORE any module-level os.environ.get(...)
# below (DASHBOARD_URL, EVAL_DELTA env override, HF_TOKEN default, WANDB_*
# defaults). Existing process env vars take precedence so a CLI-side
# `WANDB_PROJECT=foo python train_challenger.py ...` still wins.
# ---------------------------------------------------------------------------
def _load_dotenv(path: str) -> int:
    """Tiny dotenv parser. Supports `KEY=VALUE`, `# comment`, and quoted
    values. Returns the count of newly-set vars. Existing vars are NEVER
    overwritten — CLI / shell-exported values always win.
    """
    if not os.path.isfile(path):
        return 0
    n_loaded = 0
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip()
            # Strip an optional inline comment that begins with ` #` (space+#).
            # Don't strip `#` inside quoted values.
            if not (val.startswith('"') or val.startswith("'")):
                hash_pos = val.find(" #")
                if hash_pos != -1:
                    val = val[:hash_pos].rstrip()
            # Strip surrounding quotes if matched.
            if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
                val = val[1:-1]
            if key and key not in os.environ:
                os.environ[key] = val
                n_loaded += 1
    return n_loaded


_DOTENV_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".env"
)
_DOTENV_COUNT = _load_dotenv(_DOTENV_PATH)

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

import multiprocessing as mp
import shutil
import struct
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import HfApi, snapshot_download
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

# Mining harness lives at scripts/mining/; bootstrap repo root onto sys.path
# so the active arch (chain.toml -> [arch].module) registers with HF Auto*.
# When deployed to the GPU box (run_pipeline.sh tar-pushes only this script),
# the user must `pip install -e .` the teutonic repo there so chain_config /
# archs are importable.
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
import chain_config  # noqa: E402

chain_config.load_arch()

# Chain-aware LoRA target presets. PEFT matches `target_modules` strings
# against the END of each module's qualified name; substring matches do
# NOT work (the prior `gate`/`up`/`down` defaults silently froze every
# Qwen3-MoE expert). Mirrors LORA_TARGET_PRESETS in
# scripts/training_bundle/train_lora_token_ids.py — keep in sync.
LORA_TARGET_PRESETS: dict[str, list[str]] = {
    "archs.qwen3_moe": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "archs.quasar": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "ffn.gate", "ffn.up", "ffn.down",
        "w_down_proj", "w_up_proj",
    ],
}


def default_lora_targets() -> list[str]:
    return LORA_TARGET_PRESETS.get(
        chain_config.ARCH_MODULE,
        ["q_proj", "k_proj", "v_proj", "o_proj",
         "gate_proj", "up_proj", "down_proj"],
    )


# ---------------------------------------------------------------------------
# Optional W&B (orchestrator-level only — the inner training loop reports
# its own per-step metrics directly to W&B via TrainingArguments.report_to).
# ---------------------------------------------------------------------------
try:
    import wandb  # noqa: F401  type: ignore
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [train_challenger] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_challenger")

if _DOTENV_COUNT:
    log.info("loaded %d env vars from %s (existing env vars unchanged)",
             _DOTENV_COUNT, _DOTENV_PATH)
elif os.path.isfile(_DOTENV_PATH):
    log.info(".env exists at %s but added 0 new vars (all already set in env)",
             _DOTENV_PATH)
else:
    log.info("no .env at %s — relying on shell env / CLI flags", _DOTENV_PATH)

# ---------------------------------------------------------------------------
# Defaults (mirror validator constants where applicable)
# ---------------------------------------------------------------------------
SEQ_LEN = 2048
EVAL_ALPHA = 0.001
EVAL_DELTA = 0.0025  # validator's effect floor in nats/token; see eval/torch_runner.py
LM_HEAD_CHUNK = 256
DASHBOARD_URL = os.environ.get(
    "TEUTONIC_DASHBOARD_URL",
    "https://us-east-1.hippius.com/teutonic-sn3/dashboard.json",
)
HIPPIUS_BASE = "https://s3.hippius.com/teutonic-sn3"


# ---------------------------------------------------------------------------
# Shard I/O
# ---------------------------------------------------------------------------
def parse_npy_header(raw: bytes) -> tuple[int, dict]:
    buf = io.BytesIO(raw)
    if buf.read(6) != b"\x93NUMPY":
        raise ValueError("not a .npy file")
    ver = struct.unpack("BB", buf.read(2))
    hl = struct.unpack("<H" if ver[0] == 1 else "<I",
                       buf.read(2 if ver[0] == 1 else 4))[0]
    header = eval(buf.read(hl).decode("latin1").strip())
    return buf.tell(), header


def load_shard(path: Path, seq_len: int = SEQ_LEN) -> tuple[np.ndarray, int]:
    """Shards are 1D uint32 arrays (concatenated tokens). Reshape into
    (n_sequences, seq_len) by truncating tail so it divides cleanly.
    Matches validator's slicing semantics in eval_torch.fetch_sequences.
    """
    raw = path.read_bytes()
    data_offset, header = parse_npy_header(raw)
    shape = header["shape"]
    flat = np.frombuffer(raw[data_offset:], dtype="<u4")
    if len(shape) == 1:
        n_total = shape[0]
        n_seq = n_total // seq_len
        arr = flat[: n_seq * seq_len].reshape(n_seq, seq_len)
    elif len(shape) == 2:
        n_seq, seq_len = shape
        arr = flat.reshape(n_seq, seq_len)
    else:
        raise ValueError(f"unexpected shard shape {shape}")
    return arr, seq_len


def download_shard(shard_key: str, out: Path) -> Path:
    if out.exists() and out.stat().st_size > 1024:
        log.info("shard cached: %s (%.1f GB)", out, out.stat().st_size / 1e9)
        return out
    url = f"{HIPPIUS_BASE}/{shard_key}"
    log.info("downloading %s -> %s", url, out)
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["curl", "-fsSL", "-o", str(out), url])
    return out


def fetch_manifest(cache: Path) -> dict:
    p = cache / "manifest.json"
    if not p.exists():
        url = f"{HIPPIUS_BASE}/dataset/v2/manifest.json"
        log.info("downloading manifest from %s", url)
        cache.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(["curl", "-fsSL", "-o", str(p), url])
    return json.loads(p.read_text())


# ---------------------------------------------------------------------------
# King discovery
# ---------------------------------------------------------------------------
def fetch_king() -> dict:
    import urllib.request
    log.info("fetching dashboard %s", DASHBOARD_URL)
    with urllib.request.urlopen(DASHBOARD_URL, timeout=30) as r:
        d = json.loads(r.read())
    k = d["king"]
    log.info("king: repo=%s revision=%s reign=%d hotkey=%s",
             k["hf_repo"], (k.get("king_revision") or "HEAD")[:12],
             k.get("reign_number", 0), k.get("hotkey", "?")[:16])
    return k


def sha256_dir(path: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(path.glob("*.safetensors")):
        with open(p, "rb") as f:
            while chunk := f.read(1 << 20):
                h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Paired eval (mirrors eval_torch.compute_paired_losses + bootstrap)
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_per_seq_loss(model, token_batches, device, chunk=LM_HEAD_CHUNK):
    """Average per-token cross-entropy per sequence (matches eval_torch)."""
    input_ids = torch.tensor(token_batches, dtype=torch.long, device=device)
    # Reset stateful arch (Quasar latent memory) before each batch — see
    # eval_torch.compute_paired_losses for rationale. No-op for stock HF archs.
    if hasattr(model, "reset_state"):
        model.reset_state()
    out = model.model(input_ids)
    hidden = out.last_hidden_state
    lm_head = model.lm_head
    n_pos = input_ids.size(1) - 1
    total = torch.zeros(len(token_batches), device=device)
    for i in range(0, n_pos, chunk):
        end = min(i + chunk, n_pos)
        logits = lm_head(hidden[:, i:end, :])
        labels = input_ids[:, i + 1:end + 1]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            reduction="none",
        )
        total += loss.reshape(len(token_batches), -1).sum(dim=1)
        del logits, loss
    return (total / n_pos).cpu().tolist()


def paired_eval(king_dir: str, chall_dir: str, shard: np.ndarray,
                indices: list[int], device: str, batch_size: int = 8,
                n_bootstrap: int = 10000, alpha: float = EVAL_ALPHA,
                wandb_iter: int | None = None,
                chall_device: str | None = None) -> dict:
    """Mirrors validator's paired bootstrap test.

    Acceptance floor delta = EVAL_DELTA (default 0.0025 nats/token) matches
    the validator's restored fixed-effect-floor rule.

    Memory note: Qwen3-MoE in bf16 is ~155 GiB, so loading king + challenger
    on the *same* GPU OOMs even on 192 GiB B200s. If a second CUDA device is
    available we place king on `device` and challenger on `chall_device`
    (defaults to `cuda:1` when `device == cuda:0`). With only one visible
    GPU we fall back to a sequential-load strategy: score king first, evict,
    then load + score challenger.

    Intermediate mu_hat samples + final verdict are logged to W&B (if a
    run is active) under `paired_eval/iter_{wandb_iter}/...` so you can
    watch the offline test converge live.
    """
    wb_run = None
    if _HAS_WANDB:
        import wandb as _wb
        wb_run = _wb.run
    wb_prefix = f"paired_eval/iter_{wandb_iter}" if wandb_iter is not None else "paired_eval"

    # Resolve placement. Two-GPU path is strongly preferred (interleaved
    # scoring keeps both GPUs warm and halves wall-clock). One-GPU path is
    # the safety net for smoke tests on a single card.
    n_cuda = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if chall_device is None:
        if device.startswith("cuda") and n_cuda >= 2:
            king_idx = int(device.split(":")[1]) if ":" in device else 0
            chall_idx = (king_idx + 1) % n_cuda
            chall_device = f"cuda:{chall_idx}"
        else:
            chall_device = device
    two_gpu = (chall_device != device)

    delta = EVAL_DELTA

    def _score_pair(king_model, chall_model) -> tuple[np.ndarray, float, float, int]:
        diffs_l: list[float] = []
        king_sum_l = chall_sum_l = 0.0
        n_done_l = 0
        t0_local = time.time()
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i + batch_size]
            toks = [shard[j].tolist() for j in batch_idx]
            kl = compute_per_seq_loss(king_model, toks, device)
            cl = compute_per_seq_loss(chall_model, toks, chall_device)
            for k, c in zip(kl, cl):
                diffs_l.append(k - c)
                king_sum_l += k
                chall_sum_l += c
                n_done_l += 1
            if (i // batch_size) % 5 == 0:
                mu = float(np.mean(diffs_l))
                log.info("eval %d/%d | mu_hat=%.6f | king=%.4f chall=%.4f | %.1fs",
                         n_done_l, len(indices), mu,
                         king_sum_l / n_done_l, chall_sum_l / n_done_l,
                         time.time() - t0_local)
                if wb_run is not None:
                    wb_run.log({
                        f"{wb_prefix}/progress_mu_hat": mu,
                        f"{wb_prefix}/progress_king_loss": king_sum_l / n_done_l,
                        f"{wb_prefix}/progress_chall_loss": chall_sum_l / n_done_l,
                        f"{wb_prefix}/n_done": n_done_l,
                    })
        return np.asarray(diffs_l, dtype=np.float64), king_sum_l, chall_sum_l, n_done_l

    if two_gpu:
        log.info("paired_eval: loading king %s on %s", king_dir, device)
        king = AutoModelForCausalLM.from_pretrained(
            king_dir, torch_dtype=torch.bfloat16, device_map={"": device},
            use_safetensors=True,
        )
        king.eval()
        log.info("paired_eval: loading challenger %s on %s", chall_dir, chall_device)
        chall = AutoModelForCausalLM.from_pretrained(
            chall_dir, torch_dtype=torch.bfloat16, device_map={"": chall_device},
            use_safetensors=True,
        )
        chall.eval()
        t0 = time.time()
        diffs, king_sum, chall_sum, n_done = _score_pair(king, chall)
        del king, chall
        torch.cuda.empty_cache()
    else:
        # Sequential-load fallback: score king, evict, score challenger,
        # then pair them up by index. Doubles wall-clock vs the two-GPU
        # path but works on a single card.
        log.warning(
            "paired_eval: only one CUDA device visible — falling back to "
            "sequential king→challenger scoring. Expect ~2x wall-clock vs "
            "the two-GPU path."
        )
        log.info("paired_eval: loading king %s on %s", king_dir, device)
        king = AutoModelForCausalLM.from_pretrained(
            king_dir, torch_dtype=torch.bfloat16, device_map={"": device},
            use_safetensors=True,
        )
        king.eval()
        king_losses: list[float] = []
        t0 = time.time()
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i + batch_size]
            toks = [shard[j].tolist() for j in batch_idx]
            king_losses.extend(compute_per_seq_loss(king, toks, device))
        del king
        torch.cuda.empty_cache()

        log.info("paired_eval: loading challenger %s on %s", chall_dir, device)
        chall = AutoModelForCausalLM.from_pretrained(
            chall_dir, torch_dtype=torch.bfloat16, device_map={"": device},
            use_safetensors=True,
        )
        chall.eval()
        chall_losses: list[float] = []
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i + batch_size]
            toks = [shard[j].tolist() for j in batch_idx]
            chall_losses.extend(compute_per_seq_loss(chall, toks, device))
        del chall
        torch.cuda.empty_cache()

        diffs = np.asarray(king_losses, dtype=np.float64) - np.asarray(chall_losses, dtype=np.float64)
        king_sum = float(np.sum(king_losses))
        chall_sum = float(np.sum(chall_losses))
        n_done = len(diffs)

    mu_hat = float(diffs.mean())
    boot = np.empty(n_bootstrap)
    rng = np.random.default_rng(0xB007)
    for b in range(n_bootstrap):
        boot[b] = diffs[rng.integers(0, len(diffs), size=len(diffs))].mean()
    lcb = float(np.quantile(boot, alpha))
    # Symmetric upper bound for context (not used in acceptance but useful
    # for tracking how much headroom you have).
    ucb = float(np.quantile(boot, 1.0 - alpha))
    se_mu = float(diffs.std(ddof=1) / math.sqrt(len(diffs)))
    accepted = lcb > delta
    margin = mu_hat - delta  # how far you cleared the validator's floor
    res = {
        "n_eval": n_done,
        "mu_hat": mu_hat,
        "lcb": lcb,
        "ucb": ucb,
        "se_mu": se_mu,
        "margin_over_delta": margin,
        "delta": delta,
        "alpha": alpha,
        "accepted": accepted,
        "avg_king_loss": king_sum / n_done,
        "avg_chall_loss": chall_sum / n_done,
        "elapsed_s": time.time() - t0,
    }
    log.info("paired_eval: mu_hat=%.6f lcb=%.6f (delta=%.6f, margin=%+.6f) accepted=%s",
             mu_hat, lcb, delta, margin, accepted)
    if wb_run is not None:
        wb_run.log({f"{wb_prefix}/{k}": v for k, v in res.items()
                    if isinstance(v, (int, float, bool))})
    return res


# ---------------------------------------------------------------------------
# Sample scoring + curriculum (single-GPU; lifted from training_bundle)
# ---------------------------------------------------------------------------
def score_and_curate(king_dir: str, shards: list[np.ndarray],
                     n_score: int, train_per_iter: int, val_size: int,
                     seed: int, device: str, work: Path) -> tuple[Path, Path]:
    """Score `n_score` random samples on the king, bucket, write train/val jsonl."""
    rng = np.random.default_rng(seed)
    cands = []
    for s_idx, shard in enumerate(shards):
        if len(shard) == 0:
            continue
        n_take = max(n_score // len(shards), 32)
        idxs = rng.choice(len(shard), size=min(n_take, len(shard)), replace=False)
        for j in idxs:
            cands.append((s_idx, int(j)))
    rng.shuffle(cands)

    log.info("scoring %d samples with king on %s", len(cands), device)
    model = AutoModelForCausalLM.from_pretrained(
        king_dir, torch_dtype=torch.bfloat16, device_map={"": device},
        use_safetensors=True,
    )
    model.eval()

    rows = []
    BATCH = 8
    for i in range(0, len(cands), BATCH):
        chunk = cands[i:i + BATCH]
        toks = [shards[s][j].tolist() for s, j in chunk]
        losses = compute_per_seq_loss(model, toks, device)
        for (s_idx, j), tok, loss in zip(chunk, toks, losses):
            arr = np.asarray(tok)
            unique_r = float(len(set(tok)) / len(tok))
            rep_r = float(np.mean(arr[1:] == arr[:-1])) if len(arr) > 1 else 0.0
            ngrams = [tuple(tok[k:k + 4]) for k in range(len(tok) - 3)]
            rep_ng = 1.0 - len(set(ngrams)) / len(ngrams) if ngrams else 0.0
            rows.append({
                "shard": s_idx,
                "idx": j,
                "loss": float(loss),
                "unique_r": unique_r,
                "rep_r": rep_r,
                "rep_ng4": rep_ng,
                "tokens": tok,
            })

    del model
    torch.cuda.empty_cache()

    losses = np.asarray([r["loss"] for r in rows])
    p50 = float(np.percentile(losses, 50))
    p85 = float(np.percentile(losses, 85))

    def bucket(r):
        if r["rep_r"] > 0.2 or r["rep_ng4"] > 0.5 or r["unique_r"] < 0.05:
            return "suspicious"
        if r["loss"] >= p85:
            return "hard"
        if r["loss"] >= p50 * 0.8:
            return "general"
        return "easy"

    for r in rows:
        r["bucket"] = bucket(r)
    counts = {b: sum(1 for r in rows if r["bucket"] == b)
              for b in ("general", "hard", "easy", "suspicious")}
    log.info("scoring done: p50=%.3f p85=%.3f buckets=%s", p50, p85, counts)

    clean = [r for r in rows if r["bucket"] != "suspicious"]
    rng2 = np.random.default_rng(seed + 1)
    rng2.shuffle(clean)
    val_rows = clean[:val_size]
    val_keys = {(r["shard"], r["idx"]) for r in val_rows}
    pool = [r for r in clean if (r["shard"], r["idx"]) not in val_keys]

    general = [r for r in pool if r["bucket"] == "general"]
    hard = [r for r in pool if r["bucket"] == "hard"]
    easy = [r for r in pool if r["bucket"] == "easy"]
    # Curriculum mix tilted harder than the validator's training mix.
    # Rationale: the king has already learned `general` well — every gradient
    # step on a `general` sample teaches the LoRA marginally less than a
    # step on a `hard` sample. With delta=0.0025 we need data efficiency,
    # and `hard` buckets (loss >= p85) are where the king is most
    # uncertain → most room for the challenger to improve.
    # 50/40/10 was tuned empirically; bump `hard` further at the cost of
    # increased variance.
    n_general = int(train_per_iter * 0.5)
    n_hard = int(train_per_iter * 0.4)
    n_easy = train_per_iter - n_general - n_hard

    train_rows = []
    for src, n in ((general, n_general), (hard, n_hard), (easy, n_easy)):
        if not src:
            continue
        if n >= len(src):
            train_rows.extend(src)
        else:
            sel = rng2.choice(len(src), size=n, replace=False)
            train_rows.extend(src[int(k)] for k in sel)
    rng2.shuffle(train_rows)

    work.mkdir(parents=True, exist_ok=True)
    train_p = work / "train.jsonl"
    val_p = work / "val.jsonl"
    eval_p = work / "eval_indices.json"

    with open(train_p, "w") as f:
        for r in train_rows:
            f.write(json.dumps({"input_ids": r["tokens"]}) + "\n")
    with open(val_p, "w") as f:
        for r in val_rows:
            f.write(json.dumps({"input_ids": r["tokens"]}) + "\n")
    json.dump({"counts": counts, "p50": p50, "p85": p85,
               "train": len(train_rows), "val": len(val_rows)},
              open(work / "scoring.json", "w"), indent=2)
    log.info("wrote train=%d val=%d -> %s", len(train_rows), len(val_rows), work)
    return train_p, val_p


# ---------------------------------------------------------------------------
# Subprocess isolation for GPU-heavy steps
# ---------------------------------------------------------------------------
# Background: Python cannot fully release a CUDA context without exiting the
# process. `del model; torch.cuda.empty_cache()` returns *cached* memory to
# PyTorch's allocator, but the CUDA context's reserved VRAM stays attached
# to the process until it exits. If the orchestrator loads the king on
# cuda:0 for scoring (~155 GiB), then spawns `torchrun` for the inner
# trainer, the parent process keeps holding 155 GiB on cuda:0 while waiting
# on the subprocess — and rank 0 of the inner trainer OOMs trying to load
# its own king on the same GPU.
#
# Fix: spawn each GPU-heavy phase in a child process. When the child
# exits, the OS releases its CUDA context unconditionally. Workers must be
# module-level (so spawn-pickling can find them by name).
# ---------------------------------------------------------------------------
def _score_worker(king_dir: str, shard_paths: list[str], n_score: int,
                  train_per_iter: int, val_size: int, seed: int,
                  device: str, work_str: str) -> None:
    """Spawn-target for score_and_curate. Loads shards from disk and runs
    the in-process scoring. Exit code 0 on success, non-zero on failure."""
    shards = [load_shard(Path(p))[0] for p in shard_paths]
    score_and_curate(king_dir, shards, n_score, train_per_iter, val_size,
                     seed, device, Path(work_str))


def score_and_curate_isolated(king_dir: str, shard_paths: list[Path],
                               n_score: int, train_per_iter: int,
                               val_size: int, seed: int, device: str,
                               work: Path) -> tuple[Path, Path]:
    """CUDA-safe wrapper around score_and_curate. Runs the scoring in a
    fresh Python process so the king's ~155 GiB CUDA context is fully
    released when scoring finishes (before torchrun spawns)."""
    ctx = mp.get_context("spawn")
    p = ctx.Process(
        target=_score_worker,
        args=(king_dir, [str(sp) for sp in shard_paths], n_score,
              train_per_iter, val_size, seed, device, str(work)),
    )
    log.info("[isolate] scoring in subprocess (device=%s)", device)
    p.start()
    p.join()
    if p.exitcode != 0:
        raise RuntimeError(
            f"score_and_curate subprocess failed with exit code {p.exitcode}; "
            f"see traceback above")
    log.info("[isolate] scoring subprocess released CUDA context cleanly")
    return work / "train.jsonl", work / "val.jsonl"


def _paired_eval_worker(q: "mp.Queue", king_dir: str, chall_dir: str,
                         eval_shard_path: str, eval_indices: list[int],
                         device: str, batch_size: int, n_bootstrap: int,
                         alpha: float, wandb_iter: int | None) -> None:
    """Spawn-target for paired_eval. Returns the verdict dict via queue.
    Exceptions are serialized into the queue so the parent can re-raise."""
    try:
        eval_arr, _ = load_shard(Path(eval_shard_path))
        verdict = paired_eval(
            king_dir, chall_dir, eval_arr, eval_indices, device,
            batch_size=batch_size, n_bootstrap=n_bootstrap, alpha=alpha,
            wandb_iter=wandb_iter,
        )
        q.put(("ok", verdict))
    except Exception as e:  # noqa: BLE001
        q.put(("err", f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))


def paired_eval_isolated(king_dir: str, chall_dir: str,
                          eval_shard_path: Path, eval_indices: list[int],
                          device: str, batch_size: int = 8,
                          n_bootstrap: int = 10000,
                          alpha: float = EVAL_ALPHA,
                          wandb_iter: int | None = None) -> dict:
    """CUDA-safe wrapper around paired_eval. The king (~155 GiB) and
    challenger (~155 GiB) live on cuda:0 and cuda:1 during eval; when the
    subprocess exits, ~310 GiB across both GPUs is released so the next
    iteration's scoring can start clean."""
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(
        target=_paired_eval_worker,
        args=(q, king_dir, chall_dir, str(eval_shard_path), eval_indices,
              device, batch_size, n_bootstrap, alpha, wandb_iter),
    )
    log.info("[isolate] paired_eval in subprocess (king=%s, chall on cuda:1 if available)",
             device)
    p.start()
    status, payload = q.get()  # blocks until worker reports
    p.join()
    if status != "ok":
        raise RuntimeError(f"paired_eval subprocess failed:\n{payload}")
    log.info("[isolate] paired_eval subprocess released CUDA context cleanly")
    return payload


# ---------------------------------------------------------------------------
# Multi-GPU LoRA training (delegated to torchrun)
# ---------------------------------------------------------------------------
def run_lora_training(base_model: str, train_p: Path, val_p: Path,
                      out_dir: Path, n_gpus: int, args: argparse.Namespace,
                      bundle: Path, iter_idx: int,
                      resume_adapter: Path | None = None) -> Path:
    """Spawn torchrun on the inner training script.

    Forwards the optimizer / scheduler / W&B / LoRA-target knobs so the
    inner loop's TrainingArguments + LoraConfig pick them up. The inner
    script reports per-step metrics directly to W&B (under the same run if
    we initialize one).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    targets_csv = ",".join(args.lora_targets) if args.lora_targets else ""

    cmd = [
        "torchrun", f"--nproc_per_node={n_gpus}",
        str(bundle / "train_lora_token_ids.py"),
        "--base-model", base_model,
        "--train-data", str(train_p),
        "--val-data", str(val_p),
        "--output-dir", str(out_dir),
        "--seq-len", str(SEQ_LEN),
        "--micro-batch-size", str(args.micro_batch),
        "--grad-accum", str(args.grad_accum),
        "--learning-rate", str(args.lr),
        "--epochs", str(args.epochs),
        "--warmup-ratio", str(args.warmup_ratio),
        "--weight-decay", str(args.weight_decay),
        "--max-grad-norm", str(args.max_grad_norm),
        "--adam-beta1", str(args.adam_beta1),
        "--adam-beta2", str(args.adam_beta2),
        "--lr-scheduler-type", args.lr_scheduler,
        "--optim", args.optim,
        "--lora-r", str(args.lora_r),
        "--lora-alpha", str(args.lora_alpha),
        "--lora-dropout", str(args.lora_dropout),
        "--logging-steps", str(args.logging_steps),
        "--eval-steps", str(args.eval_steps_inner),
        "--save-steps", str(args.save_steps_inner),
    ]
    if targets_csv:
        cmd += ["--lora-target-modules", targets_csv]
    if args.lora_rslora:
        cmd += ["--lora-rslora"]
    if resume_adapter is not None and resume_adapter.exists():
        cmd += ["--resume-adapter", str(resume_adapter)]
        log.info("warm-starting iter %d from %s", iter_idx, resume_adapter)
    if args.wandb_project:
        cmd += ["--wandb-project", args.wandb_project]
        run_name = args.wandb_run_name or f"iter{iter_idx:02d}"
        cmd += ["--wandb-run-name", f"{run_name}-train"]
        if args.wandb_tags:
            cmd += ["--wandb-tags", args.wandb_tags]

    log.info("training: %s", " ".join(cmd))
    t0 = time.time()
    subprocess.check_call(cmd)
    log.info("training done in %.1fs", time.time() - t0)

    adapter = out_dir / "best_adapter"
    if not adapter.exists():
        # Trainer.save_model may have only put the adapter in the root output_dir.
        if (out_dir / "adapter_model.safetensors").exists() or \
           (out_dir / "adapter_model.bin").exists():
            adapter = out_dir
        else:
            raise RuntimeError(f"no adapter found in {out_dir}")
    return adapter


def merge_lora(base_model: str, adapter: Path, out: Path) -> Path:
    log.info("merging LoRA %s into %s -> %s", adapter, base_model, out)
    from peft import PeftModel
    base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, use_safetensors=True,
    )
    merged = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload()
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out), safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    tok.save_pretrained(str(out))
    # Copy config files for parity with king
    for name in ("config.json",):
        src = Path(snapshot_download(base_model, allow_patterns=[name])) / name
        if src.exists():
            shutil.copy(src, out / name)
    del base, merged
    torch.cuda.empty_cache()
    log.info("merged model saved to %s", out)
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    # I/O + iteration topology
    ap.add_argument("--work", default="/root/teutonic-mining/work",
                    help="Working dir on this box")
    ap.add_argument("--bundle", default="/root/teutonic-mining/bundle",
                    help="Path to training_bundle directory")
    ap.add_argument("--n-shards", type=int, default=2,
                    help="Number of dataset shards to download for training")
    ap.add_argument("--shard-start", type=int, default=0,
                    help="Index of first shard to use (other than eval shard)")
    ap.add_argument("--eval-shard", type=int, default=10,
                    help="Held-out shard index for offline paired eval")
    ap.add_argument("--reuse-king", action="store_true",
                    help="If a previously-downloaded king at $WORK/king matches "
                         "the live king (same repo + revision), reuse it "
                         "instead of re-downloading ~155 GiB. Falls back to "
                         "fresh download if the king has flipped on chain or "
                         "the local dir is incomplete. Dataset shards and "
                         "manifest are always cached regardless of this flag.")
    ap.add_argument("--n-eval", type=int, default=2000,
                    help="Sequences for offline paired eval (validator uses 20k)")
    ap.add_argument("--n-score", type=int, default=4000)
    ap.add_argument("--train-per-iter", type=int, default=4000)
    ap.add_argument("--val-size", type=int, default=400)
    ap.add_argument("--max-iters", type=int, default=3,
                    help="Retry training with new seed if first attempt insufficient")
    ap.add_argument("--target-mu", type=float, default=0.05,
                    help="Stop training as soon as offline mu_hat exceeds this")
    ap.add_argument("--target-lcb", type=float, default=EVAL_DELTA + 0.0001,
                    help="Stop early when LCB clears this (default just above "
                         "validator delta, so essentially `accepted=True`).")
    ap.add_argument("--warm-start-iters", action="store_true",
                    help="Each iteration after the first warm-starts from the "
                         "previous best LoRA adapter instead of re-initing.")
    # Optimizer / scheduler / training loop
    ap.add_argument("--micro-batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--adam-beta1", type=float, default=0.9)
    ap.add_argument("--adam-beta2", type=float, default=0.95)
    ap.add_argument("--lr-scheduler", default="cosine",
                    choices=["linear", "cosine", "cosine_with_restarts",
                             "polynomial", "constant", "constant_with_warmup"])
    ap.add_argument("--optim", default="adamw_torch_fused",
                    choices=["adamw_torch", "adamw_torch_fused",
                             "paged_adamw_8bit", "paged_adamw_32bit",
                             "adafactor"])
    # LoRA
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.0,
                    help="MUST be 0.0 for Qwen3-MoE / any chain whose target "
                         "modules are stored as `nn.Parameter` (PEFT's "
                         "ParamWrapper rejects non-zero dropout).")
    ap.add_argument("--lora-rslora", action="store_true",
                    help="Rank-stabilized LoRA (alpha / sqrt(r)). Recommended "
                         "when bumping --lora-r above ~32.")
    ap.add_argument("--lora-targets", default="",
                    help="Comma-separated module names to LoRA-target. "
                         "Default is chain-aware (Qwen3-MoE: q/k/v/o + "
                         "gate/up/down_proj). Pass to override.")
    # Inner trainer logging cadence
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--eval-steps-inner", type=int, default=50)
    ap.add_argument("--save-steps-inner", type=int, default=100)
    # Misc
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-gpus", type=int, default=8)
    ap.add_argument("--upload-repo", default="",
                    help="If set + accepted, push merged model to this HF repo. "
                         "Must contain the first 8 ss58 chars of your coldkey "
                         "(case-insensitive substring) or the validator will "
                         "reject the eval with `coldkey_required`. See "
                         "submit_challenger.py for the gate that enforces this.")
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    ap.add_argument("--report-out", default="",
                    help="Write a final JSON verdict to this path")
    # W&B
    ap.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", ""),
                    help="If set, init a W&B run; per-iter verdicts and inner "
                         "training metrics are logged to it.")
    ap.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", ""))
    ap.add_argument("--wandb-run-name", default="",
                    help="W&B run name. Defaults to "
                         "`<chain.NAME>-<n_gpus>g-r<lora_r>-lr<lr>-<seed>`.")
    ap.add_argument("--wandb-tags", default="",
                    help="Comma-separated tags for the W&B run.")
    args = ap.parse_args()

    # Resolve LoRA targets (comma string -> list, with chain-aware default).
    args.lora_targets = (
        [s.strip() for s in args.lora_targets.split(",") if s.strip()]
        if args.lora_targets
        else default_lora_targets()
    )
    log.info("lora targets (chain=%s, arch=%s): %s",
             chain_config.NAME, chain_config.ARCH_MODULE, args.lora_targets)

    # ----------------------------------------------------------------- W&B
    wb_run = None
    if args.wandb_project:
        if not _HAS_WANDB:
            log.error("--wandb-project set but `wandb` is not installed; "
                      "`pip install wandb` and rerun.")
            sys.exit(2)
        import wandb
        run_name = (args.wandb_run_name
                    or f"{chain_config.NAME}-{args.n_gpus}g-r{args.lora_r}"
                       f"-lr{args.lr:.0e}-s{args.seed}")
        tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
        tags = list({*tags, chain_config.NAME, chain_config.ARCH_MODULE})
        wb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=run_name,
            tags=tags,
            config={k: v for k, v in vars(args).items()
                    if k not in {"hf_token"}},
            reinit=True,
        )
        log.info("wandb run: %s", wb_run.url if wb_run else "?")

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    cache = work / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    # 1. king
    king = fetch_king()
    king_dir = work / "king"
    king_meta_path = king_dir / ".king_meta.json"
    want_repo = king["hf_repo"]
    want_rev = king.get("king_revision") or ""

    # Decide: reuse the cached king or wipe + re-download.
    # The marker file is written LAST during a successful download, so its
    # mere existence is a strong proof that the previous download finished.
    reuse_ok = False
    if args.reuse_king and king_meta_path.is_file():
        try:
            meta = json.loads(king_meta_path.read_text())
            if meta.get("hf_repo") == want_repo and (
                    not want_rev or meta.get("king_revision") == want_rev):
                reuse_ok = True
        except Exception as e:  # noqa: BLE001
            log.warning("--reuse-king: cannot parse %s (%s); re-downloading",
                        king_meta_path, e)

    if reuse_ok:
        log.info("--reuse-king: cached king matches live (%s @ %s); "
                 "skipping ~155 GiB download",
                 want_repo, (want_rev or "HEAD")[:12])
    else:
        if king_dir.exists():
            if args.reuse_king:
                log.warning("--reuse-king set but cached king does not match "
                            "live king (or is incomplete); wiping and "
                            "re-downloading.")
            shutil.rmtree(king_dir)
        log.info("downloading king to %s", king_dir)
        snapshot_download(want_repo, local_dir=str(king_dir),
                          revision=want_rev or None,
                          token=args.hf_token or None, max_workers=16)
        # Persist a marker LAST so a killed download doesn't leave a stale
        # "good" marker that --reuse-king would later trust.
        king_meta_path.write_text(json.dumps({
            "hf_repo": want_repo,
            "king_revision": want_rev,
        }))

    king_hash = sha256_dir(king_dir)
    log.info("king sha256[:16]=%s", king_hash[:16])

    # 2. dataset shards.
    # NOTE: we only download + verify; arrays are NOT held in orchestrator
    # RAM. The subprocesses (score_and_curate_isolated / paired_eval_isolated)
    # re-load from disk so the orchestrator stays CPU-only between phases.
    manifest = fetch_manifest(cache)
    train_shard_idxs = list(range(args.shard_start, args.shard_start + args.n_shards))
    if args.eval_shard in train_shard_idxs:
        raise ValueError("eval_shard cannot overlap training shards")
    shard_paths: list[Path] = []
    for idx in train_shard_idxs:
        key = manifest["shards"][idx]["key"]
        path = cache / Path(key).name
        download_shard(key, path)
        shard_paths.append(path)
        log.info("training shard %d ready at %s", idx, path)

    eval_key = manifest["shards"][args.eval_shard]["key"]
    eval_path = cache / Path(eval_key).name
    download_shard(eval_key, eval_path)
    # Load eval shard JUST to get the sequence count for index sampling,
    # then immediately drop it. The paired_eval subprocess will re-load.
    _eval_arr_tmp, _ = load_shard(eval_path)
    eval_count = len(_eval_arr_tmp)
    del _eval_arr_tmp
    rng_eval = np.random.default_rng(0xE1A)
    eval_indices = rng_eval.choice(
        eval_count, size=min(args.n_eval, eval_count), replace=False,
    ).tolist()
    log.info("held-out eval shard %d: %d sequences (sampling %d)",
             args.eval_shard, eval_count, len(eval_indices))

    best = None
    best_adapter: Path | None = None
    history = []
    for it in range(args.max_iters):
        log.info("=" * 60)
        log.info("=== iteration %d/%d ===", it + 1, args.max_iters)
        log.info("=" * 60)
        seed = args.seed + 1000 * it

        # 3+4. score+curate (in a subprocess so the king's ~155 GiB CUDA
        # context is released before torchrun spawns — otherwise rank 0 of
        # the inner trainer OOMs on cuda:0)
        iter_work = work / f"iter_{it:02d}"
        iter_work.mkdir(exist_ok=True)
        train_p, val_p = score_and_curate_isolated(
            str(king_dir), shard_paths, args.n_score,
            args.train_per_iter, args.val_size, seed, "cuda:0", iter_work,
        )

        # 5. LoRA train (optionally warm-start from previous best adapter)
        out_dir = iter_work / "lora_out"
        resume = best_adapter if (args.warm_start_iters and best_adapter) else None
        adapter = run_lora_training(
            str(king_dir), train_p, val_p, out_dir, args.n_gpus, args,
            Path(args.bundle), iter_idx=it, resume_adapter=resume,
        )

        # 6. merge
        merged_dir = iter_work / "merged"
        merge_lora(str(king_dir), adapter, merged_dir)

        # 7. paired eval (in a subprocess so king + challenger ~310 GiB
        # across cuda:0+cuda:1 is released before the next iter's scoring)
        verdict = paired_eval_isolated(
            str(king_dir), str(merged_dir), eval_path, eval_indices, "cuda:0",
            wandb_iter=it,
        )
        verdict["iter"] = it
        verdict["seed"] = seed
        history.append(verdict)
        json.dump(verdict, open(iter_work / "verdict.json", "w"), indent=2)

        if wb_run is not None:
            wb_run.log({
                "iter/mu_hat": verdict["mu_hat"],
                "iter/lcb": verdict["lcb"],
                "iter/ucb": verdict.get("ucb", 0.0),
                "iter/se_mu": verdict.get("se_mu", 0.0),
                "iter/margin_over_delta": verdict.get("margin_over_delta", 0.0),
                "iter/avg_king_loss": verdict["avg_king_loss"],
                "iter/avg_chall_loss": verdict["avg_chall_loss"],
                "iter/accepted": int(bool(verdict["accepted"])),
                "iter/elapsed_s": verdict["elapsed_s"],
                "iter/seed": seed,
                "iter/index": it,
            })
            wb_run.summary["best_mu_hat"] = max(
                wb_run.summary.get("best_mu_hat", float("-inf")),
                verdict["mu_hat"],
            )
            wb_run.summary["best_lcb"] = max(
                wb_run.summary.get("best_lcb", float("-inf")),
                verdict["lcb"],
            )

        if best is None or verdict["mu_hat"] > best["mu_hat"]:
            best = {**verdict, "iter_dir": str(iter_work),
                    "merged_dir": str(merged_dir),
                    "adapter_dir": str(adapter)}
            best_adapter = adapter
        if verdict["accepted"] and (
            verdict["mu_hat"] >= args.target_mu
            or verdict["lcb"] >= args.target_lcb
        ):
            log.info("target reached at iter %d (mu_hat=%.6f lcb=%.6f)",
                     it, verdict["mu_hat"], verdict["lcb"])
            break

    final = {
        "king_repo": king["hf_repo"],
        "king_revision": king.get("king_revision"),
        "king_hash": king_hash,
        "best": best,
        "history": history,
        "ts": time.time(),
    }
    if args.report_out:
        Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(final, open(args.report_out, "w"), indent=2)
        log.info("wrote verdict to %s", args.report_out)

    # 8. optional upload
    if args.upload_repo and best and best["accepted"]:
        log.info("uploading %s -> %s", best["merged_dir"], args.upload_repo)
        api = HfApi(token=args.hf_token)
        api.create_repo(args.upload_repo, exist_ok=True, private=False)
        api.upload_folder(
            folder_path=best["merged_dir"],
            repo_id=args.upload_repo,
            commit_message=f"Teutonic challenger (mu_hat={best['mu_hat']:.6f})",
            allow_patterns=["*.safetensors", "config.json", "tokenizer*",
                            "special_tokens*", "generation_config.json"],
        )
        info = api.repo_info(args.upload_repo)
        final["uploaded_repo"] = args.upload_repo
        final["uploaded_revision"] = info.sha
        final["challenger_hash"] = sha256_dir(Path(best["merged_dir"]))
        if args.report_out:
            json.dump(final, open(args.report_out, "w"), indent=2)
        log.info("uploaded -> %s @ %s", args.upload_repo, info.sha[:12])
    elif args.upload_repo:
        log.warning("not uploading: best=%s", best)

    if wb_run is not None:
        if best:
            wb_run.summary["final_best_mu_hat"] = best["mu_hat"]
            wb_run.summary["final_best_lcb"] = best["lcb"]
            wb_run.summary["final_accepted"] = bool(best["accepted"])
            wb_run.summary["final_avg_king_loss"] = best["avg_king_loss"]
            wb_run.summary["final_avg_chall_loss"] = best["avg_chall_loss"]
        if final.get("uploaded_repo"):
            wb_run.summary["uploaded_repo"] = final["uploaded_repo"]
            wb_run.summary["uploaded_revision"] = final.get("uploaded_revision", "")
        wb_run.finish()

    log.info("DONE — best mu_hat=%.6f accepted=%s",
             best["mu_hat"] if best else float("nan"),
             best["accepted"] if best else False)


if __name__ == "__main__":
    main()
