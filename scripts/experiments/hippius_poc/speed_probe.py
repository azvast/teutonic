"""Standalone HTTP parallel-range downloader for Hippius speed sweeping.

No boto3 dependency; just stdlib urllib. Hits a public URL, splits into
(concurrency) byte ranges of (chunk_mib) MiB each, writes to /dev/null or
a file, reports aggregate MB/s plus per-range stats.

Usage:
    python speed_probe.py \\
        --url https://s3.hippius.com/teutonic-sn3/.../model.safetensors \\
        --concurrency 32 --chunk-mib 16 --out /dev/null
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import statistics
import sys
import threading
import time
import urllib.request
from pathlib import Path


def head_length(url: str, timeout: float = 30.0) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        cl = r.headers.get("Content-Length")
        if not cl:
            raise RuntimeError("no Content-Length on HEAD")
        return int(cl)


def fetch_range(url: str, start: int, end: int, dest_fp, lock,
                timeout: float = 120.0, max_retries: int = 5) -> tuple[float, int]:
    """Fetch bytes [start, end] into dest_fp at offset `start`. Returns (seconds, bytes)."""
    attempt = 0
    pos = start
    total = 0
    t0 = time.monotonic()
    while True:
        req = urllib.request.Request(url)
        req.add_header("Range", f"bytes={pos}-{end}")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    if dest_fp is not None:
                        with lock:
                            dest_fp.seek(pos)
                            dest_fp.write(chunk)
                    pos += len(chunk)
                    total += len(chunk)
            if pos >= end + 1:
                return time.monotonic() - t0, total
            # short read -- retry remainder.
            raise IOError(f"short read: {pos - start}/{end - start + 1}")
        except Exception as e:
            attempt += 1
            if attempt > max_retries:
                raise
            backoff = min(2 ** (attempt - 1), 8)
            sys.stderr.write(f"[range {start}-{end}] attempt {attempt}: {type(e).__name__} {e}; resuming at byte {pos} in {backoff}s\n")
            time.sleep(backoff)


def build_ranges(size: int, chunk_bytes: int) -> list[tuple[int, int]]:
    r = []
    start = 0
    while start < size:
        end = min(start + chunk_bytes, size) - 1
        r.append((start, end))
        start = end + 1
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--chunk-mib", type=int, default=16)
    ap.add_argument("--out", default="/dev/null",
                    help="destination file. /dev/null = measure raw network throughput, "
                         "not disk write. Pass a path to bench disk too.")
    ap.add_argument("--max-bytes", type=int, default=0,
                    help="If > 0, stop after downloading ~this many bytes (saves time on 50GB tests).")
    args = ap.parse_args()

    size = head_length(args.url)
    chunk_bytes = args.chunk_mib * 1024 * 1024
    if args.max_bytes > 0 and args.max_bytes < size:
        # Truncate work to first max_bytes.
        ranges = build_ranges(args.max_bytes, chunk_bytes)
        work_bytes = args.max_bytes
    else:
        ranges = build_ranges(size, chunk_bytes)
        work_bytes = size

    lock = threading.Lock()
    out_path = Path(args.out)
    if args.out == "/dev/null":
        dest_fp = None
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            f.truncate(work_bytes)
        dest_fp = open(out_path, "r+b")

    sys.stderr.write(
        f"[speed_probe] size={size} ({size/1024/1024:.2f} MiB) "
        f"work={work_bytes} ({work_bytes/1024/1024:.2f} MiB) "
        f"ranges={len(ranges)} chunk={args.chunk_mib}MiB concurrency={args.concurrency} "
        f"out={args.out}\n"
    )

    per_range_s: list[float] = []
    t0 = time.monotonic()
    try:
        with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = [ex.submit(fetch_range, args.url, s, e, dest_fp, lock) for s, e in ranges]
            for i, fut in enumerate(cf.as_completed(futs)):
                sec, n = fut.result()
                per_range_s.append(sec)
                if i and i % max(1, len(ranges) // 10) == 0:
                    elapsed = time.monotonic() - t0
                    done = sum(1 for _ in per_range_s)
                    sys.stderr.write(f"  [{elapsed:6.1f}s] {done}/{len(ranges)} ranges done\n")
    finally:
        if dest_fp is not None:
            dest_fp.close()

    elapsed = time.monotonic() - t0
    mbps = (work_bytes / (1024 * 1024)) / elapsed if elapsed > 0 else 0
    sys.stderr.write(
        f"[speed_probe] DONE concurrency={args.concurrency} chunk={args.chunk_mib}MiB "
        f"elapsed={elapsed:.2f}s throughput={mbps:.2f} MB/s "
        f"per_range_p50={statistics.median(per_range_s):.2f}s "
        f"per_range_p95={statistics.quantiles(per_range_s, n=20)[-1]:.2f}s "
        f"per_range_max={max(per_range_s):.2f}s\n"
    )
    # One-line machine-parseable result on stdout.
    print(f"concurrency={args.concurrency} chunk_mib={args.chunk_mib} elapsed_s={elapsed:.3f} mbps={mbps:.2f} bytes={work_bytes}")


if __name__ == "__main__":
    main()
