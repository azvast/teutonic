# Hippius download speed ceiling

Target: `https://s3.hippius.com/teutonic-sn3/.../model.safetensors` (2.2 GiB).

Measured from two source IPs: this local workstation and the B200 box (lium `lunar-orbit-6b`, `95.133.252.113`).

## Short answer

- **B200 box peak: ~108–135 MB/s** (aria2c `-x 8 -s 8 -k 16M`). Single high-water mark of **134.8 MB/s**. Sustained average closer to **~100 MB/s with ±30% variance** across back-to-back runs.
- **Local workstation peak: ~162 MB/s** (Python thread-pool, 32 parallel 16 MiB ranges).
- **HF baseline from same B200 box: ~180 MB/s** (CDN). Hippius is **~1.5–1.8× slower** than HF from this box.
- **Hard ceiling per source IP ≈ 8 concurrent connections.** More than that makes things worse. Each connection individually caps around ~10–15 MB/s.

## B200 box — concurrency × chunk sweep (MB/s)

Python `speed_probe.py` (pure stdlib), 1 GiB slice:

| c \\ chunk | 8 MiB | 16 MiB | 32 MiB | 64 MiB | 128 MiB |
|---|---|---|---|---|---|
| 4  | 34  | 53  | 95  | 85  | **103** |
| 8  | 68  | **108** | 69  | 58  | 48  |
| 12 | 79  | 104 | 68  | 55  | 35  |
| 16 | —   | 66  | —   | —   | —   |
| 32 | —   | 64  | —   | —   | —   |
| 64 | —   | 79  | —   | —   | —   |
| 128| —   | 86  | —   | —   | —   |

**Sweet spot: `(c=8, chunk=16 MiB)` → ~108 MB/s.** Anything over c≈8 degrades (per-range p95 latency jumps from 1.9s to 13s+, strongly suggesting per-IP connection-count throttling at the gateway).

## Tool comparison (B200, same 2.2 GiB object)

| tool | best config | best MB/s | notes |
|---|---|---|---|
| curl (single connection) | — | 28.6 | floor |
| boto3 thread pool (our `download.py` v1) | `workers=16` single-stream | 23 | pathological — no range GETs |
| Python `speed_probe` (custom range downloader) | `c=8 chunk=16MiB` | **108** | what `download.py` v2 does |
| aria2c | `-x 8 -s 8 -k 16M` | **135** (peak) / ~100 avg | native C, cleanest under congestion |
| s5cmd | `-c 8 -p 16` | 63 | surprisingly weak here; re-check separately |
| 2× aria2c processes (halves) | -x8 -s8 each | 17 | confirms limit is per-IP, not per-process |
| **HF snapshot_download** (same box) | max_workers=8 | **181** | baseline for comparison |

## Local workstation — concurrency sweep

| c | MB/s (512 MiB slice, 16 MiB chunks) |
|---|---|
| 1  | 7.4  |
| 4  | 28.4 |
| 8  | 36.0 |
| 16 | 99.8 |
| 32 | **160.1** |
| 64 | 162.9 |

Local IP scales cleanly to ~160 MB/s. The ceiling is clearly **per-source-IP**, not global.

## What actually limits us

1. **Per-source-IP connection cap ≈ 8.** Beyond that the gateway throttles (per-range latency balloons), confirmed by the "aria2c × 2 processes" test: 2× aria2c from one box together moved *less* data than 1× alone.
2. **Per-connection cap ≈ 10–15 MB/s.** Multiplied by 8 = the ~100 MB/s ceiling we observe.
3. **Gateway variance.** Same command, same box, back-to-back: 73 / 82 / 108 MB/s. Anticipate ±30% in production.

## Implications for the migration

- 50 GB Quasar download at 100 MB/s = **~8.5 min** (vs 4.6 min from HF CDN at 180 MB/s). Tolerable for a king-of-the-hill duel cadence but painful for rapid iteration.
- If the eval pipeline ever wanted N parallel challengers downloaded from the *same validator IP*, the per-IP cap means total aggregate is still ~100 MB/s; individual challenger downloads will just stretch in wall time.
- **Upload is symmetric** (we measured ~99 MB/s on the best run) — same per-IP cap applies to miners uploading.

## Concrete levers to push further (untested, ranked by expected payoff)

1. **Put a CDN in front of `s3.hippius.com`.** Cloudflare or a Varnish box with `cache-control: immutable` keyed on `{bucket}/{coldkey8}/{sha256}/...`. The prefix scheme is already content-addressed, so cache keys are stable forever. This is the HF architecture.
2. **Miner-operated reverse proxies.** Each miner runs a small HTTP cache fronting their own Hippius prefix; validators pull from the proxy (more source IPs → bypass per-IP cap, though the caches would need to prefetch from Hippius too).
3. **IPFS CID path instead of S3.** If Hippius exposes the underlying IPFS CID for an uploaded object, the validator could fetch via any IPFS gateway (`dweb.link`, `cf-ipfs.com`, or a self-hosted node). Would parallelize across the IPFS swarm rather than one S3 gateway.
4. **Regional affinity.** We confirmed `us-east-1.hippius.com` is *slower* than `s3.hippius.com` from this specific box. From a box in a different region the result could flip — worth probing once validators are deployed somewhere fixed.

## Reproducing

```bash
URL="https://s3.hippius.com/teutonic-sn3/poc/20260504T131357Z-94489ed4/challengers/deadbeef/6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933/model.safetensors"

# fastest on B200 box:
aria2c -x 8 -s 8 -k 16M -o out.safetensors "$URL"

# Python equivalent (portable, with IfMatch support for TOCTOU):
python3 scripts/experiments/hippius_poc/speed_probe.py \
  --url "$URL" --concurrency 8 --chunk-mib 16 --out out.safetensors
```
