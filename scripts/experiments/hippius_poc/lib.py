"""Shared helpers for the Hippius B200 perf benchmark.

Self-contained sandbox; does not import from miner.py / validator.py / eval/
to keep the experiment hermetically isolated from production code.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.client import Config


HIPPIUS_DEFAULT_ENDPOINT = "https://s3.hippius.com"
HIPPIUS_REGIONAL_ENDPOINT = "https://us-east-1.hippius.com"
DEFAULT_BUCKET = "teutonic-sn3"

ALLOW_PATTERNS = (
    "*.safetensors",
    "config.json",
    "tokenizer*",
    "special_tokens*",
    "generation_config.json",
    "*.json",
    "*.model",
    "*.txt",
)


def env(name: str, default: str | None = None, required: bool = False) -> str:
    v = os.environ.get(name, default)
    if required and not v:
        raise RuntimeError(f"missing env var {name}")
    return v or ""


def make_s3_client(endpoint: str | None = None):
    """Return a boto3 S3 client pointed at Hippius."""
    endpoint = endpoint or env("TEUTONIC_DS_ENDPOINT", HIPPIUS_DEFAULT_ENDPOINT)
    access_key = env("TEUTONIC_DS_ACCESS_KEY") or env("HIPPIUS_ACCESS_KEY") or env("HIPPIUS_ACCESS_KEY_ID", required=True)
    secret_key = env("TEUTONIC_DS_SECRET_KEY") or env("HIPPIUS_SECRET_KEY") or env("HIPPIUS_SECRET_ACCESS_KEY", required=True)
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
            retries={"max_attempts": 10, "mode": "standard"},
            connect_timeout=15,
            read_timeout=600,
        ),
    )


def default_transfer_config(
    multipart_threshold: int = 64 * 1024 * 1024,
    multipart_chunksize: int = 64 * 1024 * 1024,
    max_concurrency: int = 16,
) -> TransferConfig:
    return TransferConfig(
        multipart_threshold=multipart_threshold,
        multipart_chunksize=multipart_chunksize,
        max_concurrency=max_concurrency,
        use_threads=True,
    )


def sha256_dir(path: str | Path, allow_glob: tuple[str, ...] = ("*.safetensors",)) -> str:
    """Match miner.py's sha256_dir semantics by default (safetensors-only)."""
    p = Path(path)
    h = hashlib.sha256()
    files: list[Path] = []
    for pat in allow_glob:
        files.extend(p.glob(pat))
    for fp in sorted(set(files)):
        with open(fp, "rb") as f:
            while chunk := f.read(1 << 20):
                h.update(chunk)
    return h.hexdigest()


def sha256_dir_full(path: str | Path) -> dict[str, str]:
    """Per-file sha256 across the allow-listed patterns; for cross-checks."""
    p = Path(path)
    out: dict[str, str] = {}
    for fp in sorted(_iter_allowed(p)):
        h = hashlib.sha256()
        with open(fp, "rb") as f:
            while chunk := f.read(1 << 20):
                h.update(chunk)
        out[fp.relative_to(p).as_posix()] = h.hexdigest()
    return out


EXCLUDE_DIR_PARTS = ("__pycache__", ".cache", ".git")


def _iter_allowed(p: Path):
    seen: set[Path] = set()
    for pat in ALLOW_PATTERNS:
        for fp in p.rglob(pat):
            if not fp.is_file():
                continue
            if any(part in EXCLUDE_DIR_PARTS for part in fp.relative_to(p).parts):
                continue
            if fp in seen:
                continue
            seen.add(fp)
            yield fp


def list_allowed(p: str | Path) -> list[Path]:
    return sorted(_iter_allowed(Path(p)))


def total_bytes(paths) -> int:
    return sum(Path(p).stat().st_size for p in paths)


def humanize_bytes(n: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    f = float(n)
    for u in units:
        if f < 1024 or u == units[-1]:
            return f"{f:.2f} {u}"
        f /= 1024
    return f"{f:.2f} TiB"


def mbps(n_bytes: int, seconds: float) -> float:
    if seconds <= 0:
        return 0.0
    return (n_bytes / (1024 * 1024)) / seconds


# ---------------------------------------------------------------------------
# JSONL bench logger
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class BenchLogger:
    path: Path
    run_id: str
    tier: str

    def emit(self, stage: str, **fields: Any) -> None:
        rec = {
            "ts": time.time(),
            "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "run_id": self.run_id,
            "tier": self.tier,
            "stage": stage,
            **fields,
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        sys.stderr.write(f"[bench {self.tier}] {stage}: " + " ".join(
            f"{k}={fields[k]}" for k in fields
        ) + "\n")


def open_logger(run_dir: str | Path, tier: str) -> BenchLogger:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_dir.name
    return BenchLogger(path=run_dir / "bench.jsonl", run_id=run_id, tier=tier)


# ---------------------------------------------------------------------------
# Time helper
# ---------------------------------------------------------------------------

class timer:
    def __enter__(self):
        self.t0 = time.monotonic()
        return self

    def __exit__(self, *_):
        self.elapsed = time.monotonic() - self.t0


def write_json(path: str | Path, obj: Any) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, default=str))


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())
