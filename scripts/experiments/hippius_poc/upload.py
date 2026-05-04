"""Stage 2: parallel multipart upload of a local model dir to Hippius S3.

Mirrors the proposed migration's miner.py flow: hash the local dir, place
files under poc/{run_id}/challengers/{coldkey8}/{sha256}/ and record a fake
on-chain commitment payload locally.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import time
from pathlib import Path

import lib


def upload_one(s3, bucket, key, local_path, transfer_cfg):
    t0 = time.monotonic()
    s3.upload_file(
        Filename=str(local_path),
        Bucket=bucket,
        Key=key,
        Config=transfer_cfg,
    )
    head = s3.head_object(Bucket=bucket, Key=key)
    return {
        "key": key,
        "rel": Path(local_path).name,
        "size": Path(local_path).stat().st_size,
        "etag": head["ETag"].strip('"'),
        "last_modified": head["LastModified"].isoformat(),
        "t_seconds": round(time.monotonic() - t0, 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", required=True)
    ap.add_argument("--bucket", default=lib.DEFAULT_BUCKET)
    ap.add_argument("--prefix-root", required=True,
                    help="e.g. poc/{run_id}/challengers")
    ap.add_argument("--coldkey8", default="deadbeef",
                    help="8-hex prefix mimicking the on-chain coldkey-prefix rule")
    ap.add_argument("--king-hash", default="0" * 16,
                    help="dummy 16-hex king hash for the synthetic commitment")
    ap.add_argument("--multipart-threshold-mib", type=int, default=64)
    ap.add_argument("--multipart-chunksize-mib", type=int, default=64)
    ap.add_argument("--max-concurrency", type=int, default=16,
                    help="boto3 in-file thread concurrency for multipart")
    ap.add_argument("--file-parallelism", type=int, default=4,
                    help="how many files to upload in parallel")
    ap.add_argument("--endpoint", default=None)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--tier", required=True)
    args = ap.parse_args()

    log = lib.open_logger(args.run_dir, args.tier)

    src = Path(args.src_dir)
    files = lib.list_allowed(src)
    if not files:
        raise SystemExit(f"no allow-listed files in {src}")
    n_bytes = lib.total_bytes(files)

    challenger_hash = lib.sha256_dir(src)
    s3_relkey = f"{args.coldkey8}/{challenger_hash}"
    prefix = f"{args.prefix_root.rstrip('/')}/{s3_relkey}"

    s3 = lib.make_s3_client(args.endpoint)
    transfer_cfg = lib.default_transfer_config(
        multipart_threshold=args.multipart_threshold_mib * 1024 * 1024,
        multipart_chunksize=args.multipart_chunksize_mib * 1024 * 1024,
        max_concurrency=args.max_concurrency,
    )

    log.emit(
        "upload_start",
        endpoint=s3.meta.endpoint_url,
        bucket=args.bucket,
        prefix=prefix,
        bytes=n_bytes,
        bytes_h=lib.humanize_bytes(n_bytes),
        files=len(files),
        sha256=challenger_hash[:16],
        multipart_threshold_mib=args.multipart_threshold_mib,
        multipart_chunksize_mib=args.multipart_chunksize_mib,
        max_concurrency=args.max_concurrency,
        file_parallelism=args.file_parallelism,
    )

    reports: list[dict] = []
    with lib.timer() as t:
        with cf.ThreadPoolExecutor(max_workers=args.file_parallelism) as ex:
            futs = {
                ex.submit(
                    upload_one, s3, args.bucket,
                    f"{prefix}/{p.relative_to(src).as_posix()}",
                    p, transfer_cfg,
                ): p
                for p in files
            }
            for fut in cf.as_completed(futs):
                rep = fut.result()
                reports.append(rep)
                log.emit("upload_file", **rep)

    last_mods = sorted(r["last_modified"] for r in reports)
    upload_report = {
        "endpoint": s3.meta.endpoint_url,
        "bucket": args.bucket,
        "prefix": prefix,
        "challenger_hash": challenger_hash,
        "coldkey8": args.coldkey8,
        "total_bytes": n_bytes,
        "wall_seconds": round(t.elapsed, 3),
        "agg_mbps": round(lib.mbps(n_bytes, t.elapsed), 2),
        "files": reports,
        "last_modified_min": last_mods[0] if last_mods else None,
        "last_modified_max": last_mods[-1] if last_mods else None,
    }
    lib.write_json(Path(args.run_dir) / f"upload_report.{args.tier}.json", upload_report)

    commitment = {
        "king_hash_16": args.king_hash[:16],
        "s3_key": s3_relkey,
        "challenger_hash": challenger_hash,
        "payload": f"{args.king_hash[:16]}:{s3_relkey}:{challenger_hash}",
    }
    lib.write_json(Path(args.run_dir) / f"commitment.{args.tier}.json", commitment)

    log.emit(
        "upload_done",
        bytes=n_bytes,
        bytes_h=lib.humanize_bytes(n_bytes),
        seconds=round(t.elapsed, 3),
        mbps=round(lib.mbps(n_bytes, t.elapsed), 2),
        files=len(reports),
        last_modified_spread_s=(
            (
                _iso_to_epoch(last_mods[-1]) - _iso_to_epoch(last_mods[0])
            ) if len(last_mods) > 1 else 0
        ),
        payload_len=len(commitment["payload"]),
    )


def _iso_to_epoch(s: str) -> float:
    from datetime import datetime
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


if __name__ == "__main__":
    main()
