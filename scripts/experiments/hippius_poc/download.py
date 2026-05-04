"""Stage 3: parallel download of a Hippius prefix to a fresh local cache.

Pins per-key ETag at metadata time (ListObjectsV2), then GetObject(IfMatch=etag)
in parallel. After download, recomputes sha256_dir and asserts equality with
the on-chain `challenger_hash` from commitment.json. --negative-test flips one
ETag and asserts the gateway returns 412 (the TOCTOU primitive the migration
depends on).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import shutil
import sys
import time
from pathlib import Path

from botocore.exceptions import ClientError

import lib


class _ShortRead(Exception):
    pass


def list_pinned(s3, bucket: str, prefix: str) -> list[dict]:
    keys: list[dict] = []
    token = None
    while True:
        kw = {"Bucket": bucket, "Prefix": prefix.rstrip("/") + "/"}
        if token:
            kw["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kw)
        for obj in resp.get("Contents", []) or []:
            keys.append({
                "key": obj["Key"],
                "size": obj["Size"],
                "etag": obj["ETag"].strip('"'),
                "last_modified": obj["LastModified"].isoformat(),
            })
        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break
    return keys


def _get_range(s3, bucket: str, key: str, start: int, end: int,
               etag: str | None, dest_fp, lock,
               max_retries: int = 5, backoff_s: float = 1.0):
    """Fetch bytes [start, end] with IfMatch; retry on mid-stream disconnects.

    Resumes via Range: bytes=<new_start>-<end> after partial progress.
    """
    from botocore.exceptions import (
        ResponseStreamingError, ReadTimeoutError, ConnectionError as BotoConnectionError,
    )
    pos = start
    attempt = 0
    while True:
        kw = {
            "Bucket": bucket,
            "Key": key,
            "Range": f"bytes={pos}-{end}",
        }
        if etag:
            kw["IfMatch"] = etag
        try:
            resp = s3.get_object(**kw)
            body = resp["Body"]
            while True:
                chunk = body.read(1 << 20)
                if not chunk:
                    break
                with lock:
                    dest_fp.seek(pos)
                    dest_fp.write(chunk)
                pos += len(chunk)
            if pos >= end + 1:
                return
            # EOF before end -- retry remainder.
            raise _ShortRead(f"short read: {pos - start}/{end - start + 1} bytes")
        except (ResponseStreamingError, ReadTimeoutError, BotoConnectionError,
                ConnectionResetError, OSError, _ShortRead) as e:
            attempt += 1
            if attempt > max_retries:
                raise
            sleep_s = backoff_s * (2 ** (attempt - 1))
            sys.stderr.write(
                f"[download] range {start}-{end} key={key[-40:]} attempt {attempt}: "
                f"{type(e).__name__} resuming at byte {pos} in {sleep_s:.1f}s\n"
            )
            time.sleep(sleep_s)


def get_one(
    s3,
    bucket: str,
    item: dict,
    dest: Path,
    if_match: bool,
    range_concurrency: int = 1,
    chunk_bytes: int = 64 * 1024 * 1024,
) -> dict:
    """Parallel-range GetObject with optional IfMatch (Hippius honors it).

    For small objects, single get_object. For large objects, divide into
    range_concurrency byte ranges and download in parallel.
    """
    import threading

    t0 = time.monotonic()
    dest.parent.mkdir(parents=True, exist_ok=True)
    etag = item["etag"] if if_match else None
    size = item["size"]

    if range_concurrency <= 1 or size <= chunk_bytes:
        kw = {"Bucket": bucket, "Key": item["key"]}
        if etag:
            kw["IfMatch"] = etag
        resp = s3.get_object(**kw)
        body = resp["Body"]
        with open(dest, "wb") as f:
            while True:
                chunk = body.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
    else:
        # Pre-allocate the destination file at full size.
        with open(dest, "wb") as f:
            f.truncate(size)
        # Build ranges of ~chunk_bytes each.
        ranges = []
        start = 0
        while start < size:
            end = min(start + chunk_bytes, size) - 1
            ranges.append((start, end))
            start = end + 1
        lock = threading.Lock()
        with open(dest, "r+b") as f:
            with cf.ThreadPoolExecutor(max_workers=range_concurrency) as ex:
                futs = [
                    ex.submit(_get_range, s3, bucket, item["key"], s, e, etag, f, lock)
                    for s, e in ranges
                ]
                for fut in cf.as_completed(futs):
                    fut.result()  # surface errors

    return {
        "key": item["key"],
        "size": size,
        "t_seconds": round(time.monotonic() - t0, 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commitment", required=True,
                    help="path to commitment.{tier}.json")
    ap.add_argument("--bucket", default=lib.DEFAULT_BUCKET)
    ap.add_argument("--prefix-root", required=True,
                    help="poc/{run_id}/challengers (matches upload)")
    ap.add_argument("--out", required=True,
                    help="local destination cache dir (will be wiped)")
    ap.add_argument("--workers", type=int, default=8,
                    help="files in flight; each large file uses --range-concurrency more threads")
    ap.add_argument("--range-concurrency", type=int, default=16,
                    help="parallel range workers per large file (boto3 TransferConfig)")
    ap.add_argument("--multipart-threshold-mib", type=int, default=64)
    ap.add_argument("--multipart-chunksize-mib", type=int, default=64)
    ap.add_argument("--endpoint", default=None)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--tier", required=True)
    ap.add_argument("--negative-test", action="store_true",
                    help="Pass a wrong IfMatch and assert 412.")
    args = ap.parse_args()

    log = lib.open_logger(args.run_dir, args.tier)

    commitment = lib.read_json(args.commitment)
    s3_relkey = commitment["s3_key"]
    expected_hash = commitment["challenger_hash"]
    prefix = f"{args.prefix_root.rstrip('/')}/{s3_relkey}"

    s3 = lib.make_s3_client(args.endpoint)

    log.emit("list_start", prefix=prefix)
    pinned = list_pinned(s3, args.bucket, prefix)
    if not pinned:
        raise SystemExit(f"prefix {prefix} returned 0 objects")
    n_bytes = sum(o["size"] for o in pinned)
    last_mods = sorted(o["last_modified"] for o in pinned)
    log.emit(
        "list_done",
        keys=len(pinned),
        bytes=n_bytes,
        bytes_h=lib.humanize_bytes(n_bytes),
        last_modified_min=last_mods[0],
        last_modified_max=last_mods[-1],
    )

    if args.negative_test:
        # Pick the smallest object, flip its ETag, expect 412.
        target = min(pinned, key=lambda o: o["size"])
        bad_item = dict(target)
        bad_item["etag"] = "0" * 32  # bogus etag
        try:
            get_one(s3, args.bucket, bad_item, Path("/tmp/_neg_test.bin"),
                    if_match=True, range_concurrency=1)
        except ClientError as e:
            code = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            err_code = e.response.get("Error", {}).get("Code")
            log.emit("negative_test_412",
                     http_status=code, error_code=err_code,
                     ok=(code == 412 or err_code in ("PreconditionFailed", "412")))
            return
        else:
            log.emit("negative_test_FAIL", ok=False,
                     note="GetObject succeeded with wrong IfMatch")
            raise SystemExit(2)

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    chunk_bytes = args.multipart_chunksize_mib * 1024 * 1024
    log.emit(
        "download_start",
        workers=args.workers,
        range_concurrency=args.range_concurrency,
        chunk_mib=args.multipart_chunksize_mib,
        prefix=prefix,
        bytes=n_bytes,
    )
    results: list[dict] = []
    errors: list[dict] = []
    with lib.timer() as t:
        with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {}
            for o in pinned:
                rel = o["key"][len(prefix.rstrip("/") + "/"):]
                dest = out / rel
                futs[ex.submit(
                    get_one, s3, args.bucket, o, dest, True,
                    args.range_concurrency, chunk_bytes,
                )] = o
            for fut in cf.as_completed(futs):
                o = futs[fut]
                try:
                    res = fut.result()
                    results.append(res)
                    log.emit("download_file", key=res["key"][-80:],
                             size=res["size"],
                             t_seconds=res["t_seconds"],
                             mbps=round(lib.mbps(res["size"], res["t_seconds"]), 2))
                except ClientError as e:
                    errors.append({
                        "key": o["key"],
                        "code": e.response.get("Error", {}).get("Code"),
                        "http": e.response.get("ResponseMetadata", {}).get("HTTPStatusCode"),
                    })
    if errors:
        log.emit("download_errors", count=len(errors), sample=errors[:3])
        raise SystemExit(f"download errors: {errors[:3]}")

    log.emit("download_done",
             bytes=n_bytes,
             bytes_h=lib.humanize_bytes(n_bytes),
             seconds=round(t.elapsed, 3),
             mbps=round(lib.mbps(n_bytes, t.elapsed), 2),
             keys=len(results))

    log.emit("verify_start")
    with lib.timer() as t_h:
        downloaded_hash = lib.sha256_dir(out)
    ok = downloaded_hash == expected_hash
    log.emit("verify_done",
             ok=ok,
             expected=expected_hash[:16],
             got=downloaded_hash[:16],
             seconds=round(t_h.elapsed, 3),
             mbps=round(lib.mbps(n_bytes, t_h.elapsed), 2))
    if not ok:
        raise SystemExit("sha256 mismatch")


if __name__ == "__main__":
    main()
