#!/usr/bin/env python3
"""Sandbox soak driver for the Teutonic-LXXX 80B Qwen3MoE eval pipeline.

Runs N sequential perturb+eval iterations against an already-running
eval_server. The king is loaded ONCE on the first iteration and cached
across all subsequent ones; every iteration generates a fresh perturbed
challenger from the same king, fires /eval with a local-path
challenger_repo, and records:

  - wall (each phase: perturb, post + load, bootstrap)
  - per-GPU VRAM at start, after perturb, peak during eval, after eval
  - HF cache size at start + end of iteration
  - verdict (mu_hat, lcb, delta, accepted)
  - eval-server PID (catches a self-kill mid-soak)

After the loop, writes:
  - per-iter JSON in soak/iter-NNNN.json
  - summary CSV in soak/summary.csv
  - markdown report in soak/REPORT.md (printed to stdout too)

Designed for the LXXX sandbox box (8x B300 SXM6, 240 cores, 871 GB free
on /). Skips HF round-trip for the challenger to keep wall manageable;
eviction-path testing is a separate exercise (soak-with-uploads).

Usage:
    cd /root/teutonic
    source .venv/bin/activate
    source /root/.creds/hf_token.env
    # Pre-req: eval_server already running on :9000 with TEUTONIC_SHARD_ACROSS_GPUS=1.
    # If it's not, scripts/sandbox_smoke.sh starts it correctly.
    python scripts/sandbox_soak.py --iters 10 --noise 1e-4
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [soak] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sandbox_soak")

EVAL_URL = "http://127.0.0.1:9000"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def nvidia_smi_mem() -> dict[int, int]:
    """Return {gpu_id: mem_used_MiB} via nvidia-smi."""
    out = subprocess.check_output([
        "nvidia-smi",
        "--query-gpu=index,memory.used",
        "--format=csv,noheader,nounits",
    ], text=True)
    res = {}
    for line in out.strip().splitlines():
        idx, mem = [x.strip() for x in line.split(",")]
        res[int(idx)] = int(mem)
    return res


def disk_used_gb(path: str) -> float:
    """du -sh in GiB (so we see HF cache growth across iters)."""
    try:
        out = subprocess.check_output(["du", "-sBM", path], text=True,
                                       stderr=subprocess.DEVNULL).split()[0]
        return float(out.rstrip("M")) / 1024.0
    except Exception:
        return -1.0


def server_pid() -> int | None:
    try:
        out = subprocess.check_output(["pgrep", "-f", "uvicorn eval_server"],
                                      text=True).strip()
        return int(out.splitlines()[0]) if out else None
    except subprocess.CalledProcessError:
        return None


def server_health() -> bool:
    try:
        import urllib.request
        with urllib.request.urlopen(f"{EVAL_URL}/health", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Perturb (fast, bf16-direct)
# ---------------------------------------------------------------------------
def perturb_king(king_dir: Path, out_dir: Path, noise: float, seed: int,
                 mode: str = "noise") -> dict:
    """Build a challenger directory from `king_dir`.

    mode='noise':    full perturbation — read each safetensor shard, add
                     bf16 N(0, noise^2) noise to every float tensor, write to
                     out_dir. ~7 min/shard × 4 shards on this box, so ~28
                     min/iter. Use only when we explicitly want to measure
                     non-zero mu_hat / lcb (real-mining-like signal).
    mode='symlink':  symlink each shard to king's blob — challenger is
                     byte-identical to king. ~50ms total. Used by the
                     leak-detection soak: every iter still exercises load /
                     forward / free, and the verdict path runs to
                     completion (mu_hat ≈ 0, accepted = False), but we
                     don't burn ~5 hours per soak on noise we don't need.
    """
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    # Symlink the small files (tokenizer, config, index) — keeps the
    # challenger directory self-contained without re-copying ~12 MB of
    # tokenizer.json on every iteration.
    for p in king_dir.iterdir():
        if p.suffix == ".safetensors":
            continue
        target = out_dir / p.name
        if p.is_file():
            target.symlink_to(p.resolve())

    st_files = sorted(king_dir.glob("*.safetensors"))

    if mode == "symlink":
        log.info("symlinking %d safetensors (mode=symlink, noise unused) -> %s",
                 len(st_files), out_dir)
        t0 = time.time()
        for st_file in st_files:
            (out_dir / st_file.name).symlink_to(st_file.resolve())
        return {
            "perturb_wall_s": round(time.time() - t0, 3),
            "n_files": len(st_files),
            "n_float_tensors": 0,
            "n_other_tensors": 0,
            "mode": "symlink",
        }

    log.info("perturbing %d safetensors with bf16 noise stdev=%.3g (seed=%d) -> %s",
             len(st_files), noise, seed, out_dir)
    torch.manual_seed(seed)

    n_floats = 0
    n_other = 0
    t0 = time.time()
    for st_file in st_files:
        t_file = time.time()
        sd = load_file(str(st_file))
        out_sd: dict = {}
        for name, tensor in sd.items():
            if tensor.dtype in (torch.bfloat16, torch.float16):
                noise_t = torch.randn(tensor.shape, dtype=tensor.dtype) * noise
                out_sd[name] = tensor + noise_t
                n_floats += 1
            elif tensor.dtype == torch.float32:
                noise_t = torch.randn(tensor.shape, dtype=torch.float32) * noise
                out_sd[name] = tensor + noise_t
                n_floats += 1
            else:
                out_sd[name] = tensor
                n_other += 1
        save_file(out_sd, str(out_dir / st_file.name))
        log.info("  %s: %d perturbed (%.1fs)",
                 st_file.name, len(sd), time.time() - t_file)
    elapsed = time.time() - t0
    return {
        "perturb_wall_s": round(elapsed, 1),
        "n_files": len(st_files),
        "n_float_tensors": n_floats,
        "n_other_tensors": n_other,
        "mode": "noise",
    }


# ---------------------------------------------------------------------------
# Eval driver
# ---------------------------------------------------------------------------
def post_eval(king_repo: str, chall_repo: str, eval_n: int,
              batch_size: int, n_bootstrap: int, shard_key: str) -> str:
    """POST /eval, return eval_id."""
    import urllib.request
    body = json.dumps({
        "king_repo": king_repo,
        "challenger_repo": chall_repo,
        "block_hash": "soak",
        "hotkey": "soak",
        "shard_key": shard_key,
        "eval_n": eval_n,
        "alpha": 0.001,
        "seq_len": 2048,
        "batch_size": batch_size,
        "n_bootstrap": n_bootstrap,
    }).encode()
    req = urllib.request.Request(
        f"{EVAL_URL}/eval", data=body,
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["eval_id"]


def get_status(eval_id: str) -> dict:
    import urllib.request
    with urllib.request.urlopen(f"{EVAL_URL}/eval/{eval_id}", timeout=10) as r:
        return json.loads(r.read())


def wait_for_verdict(eval_id: str, peak_mem: dict[int, int],
                     max_wait_s: int = 1800) -> dict:
    """Poll status every 5s, refresh nvidia-smi peaks, return final status."""
    deadline = time.time() + max_wait_s
    last_phase = None
    while time.time() < deadline:
        status = get_status(eval_id)
        state = status.get("state", "?")
        # Update peak memory snapshot
        cur = nvidia_smi_mem()
        for g, m in cur.items():
            if m > peak_mem.get(g, 0):
                peak_mem[g] = m
        # Log phase changes
        prog = status.get("progress", {}) or {}
        phase = (prog.get("done"), prog.get("total"))
        if phase != last_phase:
            log.info("  state=%s done=%s/%s mu_hat=%s",
                     state, prog.get("done"), prog.get("total"),
                     prog.get("mu_hat"))
            last_phase = phase
        if state in ("completed", "failed"):
            return status
        time.sleep(5)
    raise TimeoutError(f"eval {eval_id} did not finish in {max_wait_s}s")


# ---------------------------------------------------------------------------
# Soak loop
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--noise", type=float, default=1e-4)
    ap.add_argument("--base-seed", type=int, default=42)
    ap.add_argument("--mode", choices=["noise", "symlink"], default="symlink",
                    help="symlink: chall = king bytewise (fast, leak-test only). "
                         "noise: full perturbation (slow, real-mining-like). "
                         "Default symlink for soak-10.")
    ap.add_argument("--king-repo", default="unconst/Teutonic-LXXX-mock-king",
                    help="King HF repo (must be cached locally already, served "
                         "to eval_server as a HF repo string so it hits the cache)")
    ap.add_argument("--king-cache-dir", default="/workspace/hf-cache/hub",
                    help="HF cache hub dir (HF_HOME/hub). The king must already be "
                         "snapshot-downloaded here.")
    ap.add_argument("--soak-dir", default="/workspace/soak",
                    help="Working dir for challenger workspaces (overwritten between iters)")
    ap.add_argument("--logs-dir", default="/workspace/logs/soak",
                    help="Where per-iter JSON + summary CSV land")
    ap.add_argument("--shard-key", default="dataset/lxxx-smoke/shard_smoke.npy")
    ap.add_argument("--eval-n", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    ap.add_argument("--keep-chall", action="store_true",
                    help="Don't delete challenger workspace after each iter (uses 163GB/iter)")
    args = ap.parse_args()

    # Sanity: server alive?
    if not server_health():
        log.error("eval_server at %s is not healthy. Start it first via "
                  "scripts/sandbox_smoke.sh or manually.", EVAL_URL)
        sys.exit(2)
    pid_at_start = server_pid()
    log.info("eval_server is up; PID=%s", pid_at_start)

    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    logs_dir = Path(args.logs_dir) / ts
    logs_dir.mkdir(parents=True, exist_ok=True)
    soak_root = Path(args.soak_dir)
    soak_root.mkdir(parents=True, exist_ok=True)

    # Resolve king local path from HF cache.
    log.info("resolving king local path (cache_dir=%s)", args.king_cache_dir)
    king_local = Path(snapshot_download(
        args.king_repo,
        cache_dir=args.king_cache_dir,
        local_files_only=True,
    ))
    log.info("king local: %s", king_local)
    assert king_local.exists()
    n_king_st = len(list(king_local.glob("*.safetensors")))
    assert n_king_st == 4, f"expected 4 safetensor shards, found {n_king_st}"

    # Summary CSV header
    csv_path = logs_dir / "summary.csv"
    csv_fields = [
        "iter", "seed", "perturb_s", "eval_wall_s", "verdict_state",
        "mu_hat", "lcb", "delta", "accepted",
        "avg_king_loss", "avg_chall_loss",
        "peak_vram_max_mib", "peak_vram_per_gpu",
        "hf_cache_gb_pre", "hf_cache_gb_post",
        "server_pid", "iter_total_s",
    ]
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(csv_fields)

    rows = []
    soak_t0 = time.time()
    for i in range(args.iters):
        iter_t0 = time.time()
        seed = args.base_seed + i
        chall_dir = soak_root / f"chall-{i:04d}"
        log.info("=" * 60)
        log.info("=== ITERATION %d / %d (seed=%d, chall=%s) ===",
                 i + 1, args.iters, seed, chall_dir.name)
        log.info("=" * 60)

        # Pre-iter snapshots
        cache_pre = disk_used_gb(args.king_cache_dir)
        vram_pre = nvidia_smi_mem()
        log.info("pre-iter: HF cache=%.1f GB | VRAM(MiB)=%s",
                 cache_pre, dict(sorted(vram_pre.items())))

        # 1. perturb
        perturb_info = perturb_king(king_local, chall_dir, args.noise, seed,
                                     mode=args.mode)
        vram_after_perturb = nvidia_smi_mem()
        log.info("post-perturb: VRAM(MiB)=%s",
                 dict(sorted(vram_after_perturb.items())))

        # 2. POST /eval
        eval_id = post_eval(
            king_repo=args.king_repo,
            chall_repo=str(chall_dir),
            eval_n=args.eval_n,
            batch_size=args.batch_size,
            n_bootstrap=args.n_bootstrap,
            shard_key=args.shard_key,
        )
        log.info("eval_id=%s", eval_id)

        # 3. wait for verdict (refreshing peak VRAM as we go)
        peak_mem: dict[int, int] = dict(vram_after_perturb)
        eval_t0 = time.time()
        try:
            status = wait_for_verdict(eval_id, peak_mem, max_wait_s=1800)
        except Exception as e:
            log.error("iter %d eval failed: %s", i, e)
            status = {"state": "timeout", "verdict": None, "error": str(e)}
        eval_wall = time.time() - eval_t0

        # 4. Post-iter snapshots
        vram_post = nvidia_smi_mem()
        cache_post = disk_used_gb(args.king_cache_dir)
        pid_now = server_pid()
        log.info("post-iter: VRAM(MiB)=%s | HF cache=%.1f GB | server PID=%s",
                 dict(sorted(vram_post.items())), cache_post, pid_now)

        if pid_now != pid_at_start:
            log.error("EVAL SERVER PID CHANGED %s -> %s — supervisor restart "
                      "(self-kill?) detected mid-soak", pid_at_start, pid_now)

        # 5. Persist iter record
        verdict = status.get("verdict") or {}
        row = {
            "iter": i,
            "seed": seed,
            "perturb_s": perturb_info["perturb_wall_s"],
            "eval_wall_s": round(eval_wall, 1),
            "verdict_state": status.get("state"),
            "mu_hat": verdict.get("mu_hat"),
            "lcb": verdict.get("lcb"),
            "delta": verdict.get("delta"),
            "accepted": verdict.get("accepted"),
            "avg_king_loss": verdict.get("avg_king_loss"),
            "avg_chall_loss": verdict.get("avg_challenger_loss"),
            "peak_vram_max_mib": max(peak_mem.values()) if peak_mem else 0,
            "peak_vram_per_gpu": json.dumps(dict(sorted(peak_mem.items()))),
            "hf_cache_gb_pre": round(cache_pre, 1),
            "hf_cache_gb_post": round(cache_post, 1),
            "server_pid": pid_now,
            "iter_total_s": round(time.time() - iter_t0, 1),
        }
        rows.append(row)
        (logs_dir / f"iter-{i:04d}.json").write_text(json.dumps(
            {"row": row, "perturb": perturb_info, "status": status,
             "vram_pre": vram_pre, "vram_after_perturb": vram_after_perturb,
             "vram_post": vram_post, "peak_mem": peak_mem},
            indent=2, default=str,
        ))
        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, csv_fields).writerow(row)

        log.info("iter %d done: wall=%.1fs eval=%.1fs perturb=%.1fs "
                 "mu_hat=%s lcb=%s peak_vram=%d MiB",
                 i, row["iter_total_s"], row["eval_wall_s"], row["perturb_s"],
                 row["mu_hat"], row["lcb"], row["peak_vram_max_mib"])

        # Free disk for the next iteration
        if not args.keep_chall:
            shutil.rmtree(chall_dir, ignore_errors=True)

    soak_wall = time.time() - soak_t0

    # Final report
    report = []
    report.append("# Teutonic-LXXX Soak-10 Report")
    report.append("")
    report.append(f"Run timestamp (UTC): {ts}")
    report.append(f"Iterations: {args.iters}")
    report.append(f"Total wall: {soak_wall/60:.1f} min")
    report.append(f"eval_server PID at start: {pid_at_start}")
    pids_seen = sorted({r["server_pid"] for r in rows if r["server_pid"]})
    report.append(f"eval_server PIDs across soak: {pids_seen} "
                  f"({'STABLE' if len(pids_seen) <= 1 else 'RESTART DETECTED'})")
    report.append("")
    report.append("## Per-iteration summary")
    report.append("")
    report.append("| iter | seed | perturb | eval | total | mu_hat | lcb | accepted | peak VRAM (MiB) | cache GB pre/post |")
    report.append("|---:|---:|---:|---:|---:|---:|---:|:-:|---:|:-:|")
    for r in rows:
        accepted = r["accepted"]
        accepted_s = "✓" if accepted else "✗" if accepted is False else "?"
        report.append(
            f"| {r['iter']} | {r['seed']} | {r['perturb_s']:.0f}s | "
            f"{r['eval_wall_s']:.0f}s | {r['iter_total_s']:.0f}s | "
            f"{r['mu_hat']} | {r['lcb']} | {accepted_s} | "
            f"{r['peak_vram_max_mib']} | "
            f"{r['hf_cache_gb_pre']:.0f}/{r['hf_cache_gb_post']:.0f} |"
        )
    report.append("")

    # Leak detection: peak VRAM should be ~constant across iterations
    peaks = [r["peak_vram_max_mib"] for r in rows]
    if peaks:
        report.append(f"## Leak detection")
        report.append("")
        report.append(f"- Peak VRAM range: {min(peaks)} .. {max(peaks)} MiB "
                      f"(spread {max(peaks) - min(peaks)} MiB)")
        last5 = peaks[-5:]
        first5 = peaks[:5]
        if first5 and last5:
            drift = sum(last5) / len(last5) - sum(first5) / len(first5)
            report.append(f"- Mean peak first-5 vs last-5: drift {drift:+.0f} MiB "
                          f"({'CONCERN' if abs(drift) > 5000 else 'OK'})")

    report_path = logs_dir / "REPORT.md"
    report_path.write_text("\n".join(report))
    log.info("=" * 60)
    log.info("SOAK DONE — report at %s", report_path)
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
