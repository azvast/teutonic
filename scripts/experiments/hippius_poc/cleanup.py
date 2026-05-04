"""Stage 5: delete a poc/ run prefix; --purge-stale wipes >24h orphans.

Run-prefix timestamps live in the second-level dirname: poc/{utc_iso}-{uuid8}/...
"""
from __future__ import annotations

import argparse
import re
import time
from datetime import datetime, timezone

import lib


# poc/20260504T125500Z-deadbeef/ ...
RUN_DIR_RE = re.compile(r"^poc/(\d{8}T\d{6}Z)-[0-9a-fA-F]{8}/")


def delete_prefix(s3, bucket: str, prefix: str) -> int:
    n = 0
    token = None
    while True:
        kw = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kw["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kw)
        objs = resp.get("Contents", []) or []
        if objs:
            s3.delete_objects(
                Bucket=bucket,
                Delete={"Objects": [{"Key": o["Key"]} for o in objs], "Quiet": True},
            )
            n += len(objs)
        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break
    return n


def parse_run_ts(prefix: str) -> datetime | None:
    m = RUN_DIR_RE.match(prefix)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def list_run_prefixes(s3, bucket: str, root: str = "poc/") -> list[str]:
    """Return distinct second-level prefixes (poc/<run_id>/)."""
    runs: set[str] = set()
    token = None
    while True:
        kw = {"Bucket": bucket, "Prefix": root, "Delimiter": "/"}
        if token:
            kw["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kw)
        for cp in resp.get("CommonPrefixes", []) or []:
            runs.add(cp["Prefix"])
        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break
    return sorted(runs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", default=lib.DEFAULT_BUCKET)
    ap.add_argument("--prefix", help="Single prefix to nuke (e.g. poc/20260504T125500Z-deadbeef/)")
    ap.add_argument("--purge-stale", action="store_true",
                    help="Purge any poc/<run>/ older than --max-age-hours")
    ap.add_argument("--max-age-hours", type=int, default=24)
    ap.add_argument("--endpoint", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    s3 = lib.make_s3_client(args.endpoint)

    if args.prefix:
        if args.dry_run:
            print(f"would delete prefix s3://{args.bucket}/{args.prefix}")
        else:
            n = delete_prefix(s3, args.bucket, args.prefix)
            print(f"deleted {n} objects under {args.prefix}")

    if args.purge_stale:
        cutoff = time.time() - args.max_age_hours * 3600
        runs = list_run_prefixes(s3, args.bucket)
        for r in runs:
            ts = parse_run_ts(r)
            if ts is None:
                print(f"skip (unparsable): {r}")
                continue
            age_h = (time.time() - ts.timestamp()) / 3600
            if ts.timestamp() < cutoff:
                if args.dry_run:
                    print(f"would purge {r} (age {age_h:.1f}h)")
                else:
                    n = delete_prefix(s3, args.bucket, r)
                    print(f"purged {n} objects under {r} (age {age_h:.1f}h)")
            else:
                print(f"keep {r} (age {age_h:.1f}h)")


if __name__ == "__main__":
    main()
