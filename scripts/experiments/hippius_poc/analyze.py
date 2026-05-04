"""Stage 6: read bench.jsonl, emit a markdown summary with go/no-go assessment."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


PASS_CRITERIA = {
    "upload_mbps_min": 200.0,
    "download_mbps_min": 400.0,
    "lastmodified_spread_max_s": 5.0,
}


def fmt(v, suffix="", default="n/a"):
    if v is None:
        return default
    if isinstance(v, float):
        return f"{v:.2f}{suffix}"
    return f"{v}{suffix}"


def load_records(path: Path):
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def summarize_tier(recs):
    by_stage = defaultdict(list)
    for r in recs:
        by_stage[r.get("stage", "")].append(r)

    def first(stage, key):
        if stage in by_stage and by_stage[stage]:
            return by_stage[stage][-1].get(key)
        return None

    summary = {
        "hf_dl_mbps": first("hf_dl_done", "mbps"),
        "hf_dl_bytes_h": first("hf_dl_done", "bytes_h"),
        "hf_dl_seconds": first("hf_dl_done", "seconds"),
        "upload_mbps": first("upload_done", "mbps"),
        "upload_bytes_h": first("upload_done", "bytes_h"),
        "upload_seconds": first("upload_done", "seconds"),
        "upload_lastmod_spread_s": first("upload_done", "last_modified_spread_s"),
        "download_mbps": first("download_done", "mbps"),
        "download_seconds": first("download_done", "seconds"),
        "verify_ok": first("verify_done", "ok"),
        "verify_seconds": first("verify_done", "seconds"),
        "verify_mbps": first("verify_done", "mbps"),
        "neg_test_ok": first("negative_test_412", "ok"),
        "vllm_load_seconds": first("vllm_load_done", "seconds"),
        "vllm_ttft_seconds": first("vllm_ttft_done", "seconds"),
        "vllm_tok_per_s": first("vllm_throughput_done", "tok_per_s"),
        "vllm_per_seq_tok_per_s": first("vllm_throughput_done", "per_seq_tok_per_s"),
        "hf_load_seconds": first("hf_load_done", "seconds"),
    }
    return summary


def assess(summaries: dict[str, dict]) -> list[tuple[str, bool, str]]:
    assessments = []
    for tier, s in summaries.items():
        if s["upload_mbps"] is not None:
            ok = s["upload_mbps"] >= PASS_CRITERIA["upload_mbps_min"]
            assessments.append((
                f"{tier} upload >= {PASS_CRITERIA['upload_mbps_min']} MB/s",
                ok,
                f"got {s['upload_mbps']:.2f} MB/s",
            ))
        if s["download_mbps"] is not None:
            ok = s["download_mbps"] >= PASS_CRITERIA["download_mbps_min"]
            assessments.append((
                f"{tier} download >= {PASS_CRITERIA['download_mbps_min']} MB/s",
                ok,
                f"got {s['download_mbps']:.2f} MB/s",
            ))
        if s["upload_lastmod_spread_s"] is not None:
            ok = float(s["upload_lastmod_spread_s"]) <= PASS_CRITERIA["lastmodified_spread_max_s"]
            assessments.append((
                f"{tier} LastModified spread <= {PASS_CRITERIA['lastmodified_spread_max_s']}s",
                ok,
                f"got {s['upload_lastmod_spread_s']}s",
            ))
        if s["verify_ok"] is not None:
            assessments.append((
                f"{tier} sha256 roundtrip",
                bool(s["verify_ok"]),
                "ok" if s["verify_ok"] else "MISMATCH",
            ))
        if s["neg_test_ok"] is not None:
            assessments.append((
                f"{tier} IfMatch 412 on bogus etag",
                bool(s["neg_test_ok"]),
                "ok" if s["neg_test_ok"] else "gateway accepted bogus etag",
            ))
    return assessments


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    bench = load_records(run_dir / "bench.jsonl")
    if not bench:
        print(f"# Hippius B200 bench: no records found under {run_dir}/bench.jsonl")
        return

    by_tier: dict[str, list[dict]] = defaultdict(list)
    for r in bench:
        by_tier[r.get("tier", "?")].append(r)

    summaries = {tier: summarize_tier(recs) for tier, recs in by_tier.items()}

    print(f"# Hippius B200 perf benchmark — {run_dir.name}")
    print()
    print("## Throughput / latency")
    print()
    print("| tier | HF dl | upload | download | verify | vLLM load | TTFT | tok/s | tok/s/seq |")
    print("|---|---|---|---|---|---|---|---|---|")
    for tier in sorted(summaries):
        s = summaries[tier]
        print(
            "| {tier} | {hf} | {up} | {dl} | {ver} | {vload} | {ttft} | {tps} | {ptps} |".format(
                tier=tier,
                hf=fmt(s["hf_dl_mbps"], " MB/s"),
                up=fmt(s["upload_mbps"], " MB/s"),
                dl=fmt(s["download_mbps"], " MB/s"),
                ver=fmt(s["verify_mbps"], " MB/s"),
                vload=fmt(s["vllm_load_seconds"], " s") if s["vllm_load_seconds"] else fmt(s["hf_load_seconds"], " s (hf)"),
                ttft=fmt(s["vllm_ttft_seconds"], " s"),
                tps=fmt(s["vllm_tok_per_s"]),
                ptps=fmt(s["vllm_per_seq_tok_per_s"]),
            )
        )
    print()

    print("## Volumes")
    print()
    print("| tier | HF size | HF time | upload time | download time |")
    print("|---|---|---|---|---|")
    for tier in sorted(summaries):
        s = summaries[tier]
        print(f"| {tier} | {fmt(s['hf_dl_bytes_h'])} | {fmt(s['hf_dl_seconds'], ' s')} | {fmt(s['upload_seconds'], ' s')} | {fmt(s['download_seconds'], ' s')} |")
    print()

    print("## Pass/fail vs migration criteria")
    print()
    asmt = assess(summaries)
    if not asmt:
        print("_No assessments yet (incomplete run)._")
    else:
        for label, ok, detail in asmt:
            mark = "PASS" if ok else "FAIL"
            print(f"- **{mark}** {label} — {detail}")
    print()

    overall = all(ok for _, ok, _ in asmt) if asmt else False
    print(f"## Overall: {'GO' if overall else 'NO-GO / NEEDS REVIEW'}")


if __name__ == "__main__":
    main()
