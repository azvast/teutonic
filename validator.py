#!/usr/bin/env python3
"""Teutonic validator — single-file king-of-the-hill evaluator.

Polls Bittensor chain for challenger submissions, dispatches evaluations
to a remote eval server (eval_server.py on a GPU box), manages king
lifecycle on HuggingFace, persists all state to R2.
"""
import asyncio
import hashlib
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone

import bittensor as bt
import boto3
import httpx
from botocore.config import Config as BotoConfig
from huggingface_hub import HfApi, snapshot_download

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EVAL_N = 10_000
EVAL_ALPHA = 0.001
SEQ_LEN = 2048
POLL_INTERVAL = 30
WEIGHT_INTERVAL = 360
NETUID = int(os.environ.get("TEUTONIC_NETUID", "3"))
NETWORK = os.environ.get("TEUTONIC_NETWORK", "finney")
KING_REPO = os.environ.get("TEUTONIC_KING_REPO", "unconst/Teutonic-I")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
EVAL_SERVER_URL = os.environ.get("TEUTONIC_EVAL_SERVER", "http://localhost:9000")
WALLET_NAME = os.environ.get("BT_WALLET_NAME", "teutonic")
WALLET_HOTKEY = os.environ.get("BT_WALLET_HOTKEY", "default")

R2_ENDPOINT = os.environ.get("TEUTONIC_R2_ENDPOINT", "")
R2_BUCKET = os.environ.get("TEUTONIC_R2_BUCKET", "")
R2_ACCESS_KEY = os.environ.get("TEUTONIC_R2_ACCESS_KEY", "")
R2_SECRET_KEY = os.environ.get("TEUTONIC_R2_SECRET_KEY", "")

REPO_PATTERN = r"^[^/]+/Teutonic-I-.+$"

log = logging.getLogger("teutonic")

# ---------------------------------------------------------------------------
# R2
# ---------------------------------------------------------------------------

class R2:
    def __init__(self):
        self.client = boto3.client(
            "s3", endpoint_url=R2_ENDPOINT,
            aws_access_key_id=R2_ACCESS_KEY, aws_secret_access_key=R2_SECRET_KEY,
            region_name="auto",
            config=BotoConfig(retries={"max_attempts": 3, "mode": "adaptive"}),
        )

    def put(self, key, data):
        self.client.put_object(
            Bucket=R2_BUCKET, Key=key,
            Body=json.dumps(data, default=str).encode(),
            ContentType="application/json",
        )

    def get(self, key):
        try:
            return json.loads(
                self.client.get_object(Bucket=R2_BUCKET, Key=key)["Body"].read()
            )
        except Exception:
            return None

    def append_jsonl(self, key, record):
        line = json.dumps(record, default=str) + "\n"
        existing = b""
        try:
            existing = self.client.get_object(Bucket=R2_BUCKET, Key=key)["Body"].read()
        except Exception:
            pass
        self.client.put_object(
            Bucket=R2_BUCKET, Key=key,
            Body=existing + line.encode(),
            ContentType="application/x-ndjson",
        )

    def append_jsonl_batch(self, key, records):
        lines = "".join(json.dumps(r, default=str) + "\n" for r in records)
        existing = b""
        try:
            existing = self.client.get_object(Bucket=R2_BUCKET, Key=key)["Body"].read()
        except Exception:
            pass
        self.client.put_object(
            Bucket=R2_BUCKET, Key=key,
            Body=existing + lines.encode(),
            ContentType="application/x-ndjson",
        )

    def put_raw(self, key, body, content_type):
        self.client.put_object(
            Bucket=R2_BUCKET, Key=key, Body=body, ContentType=content_type,
        )

    def range_get(self, key, start, end):
        return self.client.get_object(
            Bucket=R2_BUCKET, Key=key, Range=f"bytes={start}-{end}"
        )["Body"].read()


# ---------------------------------------------------------------------------
# Challenger validation
# ---------------------------------------------------------------------------

_king_config: dict | None = None

def get_king_config():
    """Fetch and cache the king model's config.json from HuggingFace."""
    global _king_config
    if _king_config is not None:
        return _king_config
    try:
        api = HfApi(token=HF_TOKEN or None)
        cfg_path = api.hf_hub_download(KING_REPO, "config.json", token=HF_TOKEN or None)
        with open(cfg_path) as f:
            _king_config = json.load(f)
    except Exception:
        log.warning("could not fetch king config.json")
        _king_config = {}
    return _king_config


def validate_challenger_config(hf_repo: str) -> str | None:
    """Check challenger config.json matches king architecture before deploying.

    Returns None if OK, or a human-readable rejection reason.
    """
    king_cfg = get_king_config()
    if not king_cfg:
        return None

    try:
        api = HfApi(token=HF_TOKEN or None)
        cfg_path = api.hf_hub_download(hf_repo, "config.json", token=HF_TOKEN or None)
        with open(cfg_path) as f:
            challenger_cfg = json.load(f)
    except Exception as e:
        return f"cannot fetch config.json: {e}"

    king_arch = king_cfg.get("architectures", [])
    chall_arch = challenger_cfg.get("architectures", [])
    if king_arch and chall_arch and king_arch != chall_arch:
        return f"architecture mismatch: king={king_arch} challenger={chall_arch}"

    for key in ("vocab_size", "hidden_size", "num_hidden_layers",
                "num_attention_heads", "num_key_value_heads", "head_dim",
                "intermediate_size", "model_type"):
        king_val = king_cfg.get(key)
        chall_val = challenger_cfg.get(key)
        if king_val is not None and chall_val is not None and king_val != chall_val:
            return f"{key} mismatch: king={king_val} challenger={chall_val}"

    st_files = [s for s in api.list_repo_files(hf_repo, token=HF_TOKEN or None)
                if s.endswith(".safetensors")]
    if not st_files:
        return "no .safetensors files in repo"

    return None


# ---------------------------------------------------------------------------
# Chain
# ---------------------------------------------------------------------------

import re
_REPO_RE = re.compile(REPO_PATTERN)

def scan_reveals(subtensor, netuid, seen):
    try:
        all_reveals = subtensor.get_all_revealed_commitments(netuid)
    except Exception:
        log.exception("failed to fetch reveals")
        return []
    if not all_reveals:
        return []

    new = []
    for hotkey, entries in all_reveals.items():
        if hotkey in seen or not entries:
            continue
        block, data = max(entries, key=lambda e: e[0])
        parts = data.split(":", 2)
        if len(parts) != 3:
            continue
        king_hash, hf_repo, model_hash = parts
        if not _REPO_RE.match(hf_repo.strip()):
            continue
        seen.add(hotkey)
        new.append({
            "hotkey": hotkey, "block": block,
            "king_hash": king_hash.strip(), "hf_repo": hf_repo.strip(),
            "model_hash": model_hash.strip(),
        })
    new.sort(key=lambda x: x["block"])
    return new


def set_weights(subtensor, wallet, netuid, king_hotkey):
    try:
        meta = subtensor.metagraph(netuid)
        if king_hotkey in meta.hotkeys:
            uid = meta.hotkeys.index(king_hotkey)
            subtensor.set_weights(wallet=wallet, netuid=netuid, uids=[uid], weights=[1.0])
            log.info("weights set 100%% to uid=%d (%s)", uid, king_hotkey[:16])
        else:
            log.warning("king hotkey %s not in metagraph", king_hotkey[:16])
    except Exception:
        log.exception("failed to set weights")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def _now():
    return datetime.now(timezone.utc).isoformat()


class State:
    def __init__(self, r2):
        self.r2 = r2
        self.king = {}
        self.queue = []
        self.seen = set()
        self.failed_repos: set[str] = set()
        self.evaluated_repos: set[str] = set()
        self.stats = {"queued": 0, "accepted": 0, "rejected": 0, "failed": 0}
        self.counter = 0
        self.current_eval = None
        self.history = []
        self.last_weight_block = 0
        self._metagraph = None
        self._metagraph_block = 0

    def refresh_metagraph(self, subtensor, netuid, *, max_age=120):
        try:
            block = subtensor.block
            if self._metagraph is None or block - self._metagraph_block >= max_age:
                self._metagraph = subtensor.metagraph(netuid)
                self._metagraph_block = block
        except Exception:
            log.exception("failed to refresh metagraph")

    def hotkey_to_uid(self, hotkey: str) -> int | None:
        if self._metagraph is None:
            return None
        try:
            return self._metagraph.hotkeys.index(hotkey)
        except ValueError:
            return None

    def load(self):
        k = self.r2.get("king/current.json")
        if k:
            self.king = k
        q = self.r2.get("state/queue.json")
        if q:
            self.queue = q.get("pending", [])
        s = self.r2.get("state/seen_hotkeys.json")
        if s:
            self.seen = set(s.get("hotkeys", []))
        st = self.r2.get("state/validator_state.json")
        if st:
            loaded = st.get("stats", self.stats)
            if "challenges" in loaded and "queued" not in loaded:
                loaded["queued"] = loaded.pop("challenges")
            loaded.setdefault("failed", 0)
            loaded.setdefault("queued", 0)
            self.stats = loaded
            self.counter = st.get("counter", 0)
        h = self.r2.get("state/dashboard_history.json")
        if h:
            self.history = h.get("history", [])
        log.info("loaded state: king=%s queue=%d seen=%d",
                 self.king.get("king_hash", "none")[:16], len(self.queue), len(self.seen))

    def flush(self):
        self.r2.put("state/validator_state.json", {
            "king": self.king, "queue": self.queue,
            "stats": self.stats, "counter": self.counter, "updated_at": _now(),
        })
        self.r2.put("state/queue.json", {"pending": self.queue, "updated_at": _now()})
        self.r2.put("king/current.json", self.king)
        self.r2.put("state/seen_hotkeys.json", {
            "hotkeys": sorted(self.seen), "updated_at": _now(),
        })

    def event(self, data):
        data.setdefault("timestamp", _now())
        self.r2.append_jsonl("state/history.jsonl", data)

    def next_id(self):
        self.counter += 1
        return f"eval-{self.counter:04d}"

    def enqueue(self, reveal):
        repo = reveal.get("hf_repo", "")
        for existing in self.queue:
            if existing.get("hf_repo") == repo:
                log.info("skipping duplicate repo: %s already queued", repo)
                return None
        if repo in self.evaluated_repos:
            log.info("skipping %s: already evaluated this cycle", repo)
            return None
        cid = self.next_id()
        entry = {"challenge_id": cid, **reveal, "queued_at": _now()}
        self.queue.append(entry)
        self.stats["queued"] += 1
        self.flush()
        self.flush_dashboard()
        self.event({"event": "queued", **entry})
        return cid

    def set_king(self, hotkey, hf_repo, king_hash, block, challenge_id="seed"):
        global _king_config
        _king_config = None
        self.failed_repos.clear()
        self.evaluated_repos.clear()
        reign = self.king.get("reign_number", 0) + (0 if challenge_id == "seed" else 1)
        self.king = {
            "hotkey": hotkey, "hf_repo": hf_repo, "king_hash": king_hash,
            "reign_number": reign, "crowned_at": _now(),
            "crowned_block": block, "challenge_id": challenge_id,
        }
        self.flush()
        self.flush_dashboard()
        self.event({"event": "king_changed", "hotkey": hotkey, "reign": reign,
                     "challenge_id": challenge_id})

    def record_verdict(self, verdict, challenger_repo, hotkey):
        king_loss = verdict["avg_king_loss"]
        chall_loss = verdict["avg_challenger_loss"]
        self.history.insert(0, {
            "challenge_id": verdict["challenge_id"],
            "hotkey": hotkey,
            "uid": self.hotkey_to_uid(hotkey),
            "challenger_repo": challenger_repo,
            "accepted": verdict["accepted"],
            "verdict": verdict["verdict"],
            "win_rate": verdict["win_rate"],
            "avg_king_loss": king_loss,
            "avg_challenger_loss": chall_loss,
            "best_loss": min(king_loss, chall_loss),
            "wall_time_s": verdict["wall_time_s"],
            "timestamp": verdict["timestamp"],
        })
        self.history = self.history[:50]
        self.r2.put("state/dashboard_history.json", {"history": self.history})

    def flush_dashboard(self):
        self.r2.put("dashboard.json", {
            "updated_at": _now(),
            "king": self.king,
            "stats": self.stats,
            "current_eval": self.current_eval,
            "queue": [{"challenge_id": e.get("challenge_id"), "hotkey": e.get("hotkey"),
                        "uid": self.hotkey_to_uid(e.get("hotkey", "")),
                        "hf_repo": e.get("hf_repo"), "queued_at": e.get("queued_at"),
                        "block": e.get("block")}
                       for e in self.queue],
            "history": self.history,
        })


# ---------------------------------------------------------------------------
# King management
# ---------------------------------------------------------------------------

def sha256_dir(path):
    """SHA256 over sorted safetensors files (matches validation.py)."""
    import hashlib as hl
    h = hl.sha256()
    from pathlib import Path
    for p in sorted(Path(path).glob("*.safetensors")):
        with open(p, "rb") as f:
            while chunk := f.read(1 << 20):
                h.update(chunk)
    return h.hexdigest()


def fork_winner(challenger_repo, king_hash, hotkey, challenge_id):
    """Upload challenger weights to king repo (creates git history)."""
    api = HfApi(token=HF_TOKEN)
    api.create_repo(KING_REPO, exist_ok=True, private=False)
    tmp = "/tmp/teutonic/fork"
    snapshot_download(challenger_repo, local_dir=tmp, token=HF_TOKEN or None,
                      allow_patterns=["*.safetensors"],
                      ignore_patterns=["*.bin", "*.pt", "__pycache__/*"])
    api.upload_folder(
        folder_path=tmp, repo_id=KING_REPO,
        commit_message=f"King #{king_hash[:8]} dethroned by {hotkey[:16]} ({challenge_id})",
        allow_patterns=["*.safetensors"],
    )
    new_hash = sha256_dir(tmp)
    log.info("forked %s -> %s hash=%s", challenger_repo, KING_REPO, new_hash[:16])
    return new_hash


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def process_challenge(state, r2, entry, subtensor, wallet, *, check_stale=True):
    cid = entry["challenge_id"]
    hotkey = entry["hotkey"]
    hf_repo = entry["hf_repo"]
    log.info("processing %s from %s repo=%s", cid, hotkey[:16], hf_repo)

    if hf_repo in state.failed_repos:
        log.info("skipping %s: repo %s previously failed", cid, hf_repo)
        return

    if hf_repo in state.evaluated_repos:
        log.info("skipping %s: repo %s already evaluated this cycle", cid, hf_repo)
        return

    rejection = validate_challenger_config(hf_repo)
    if rejection:
        log.warning("rejecting %s (%s): %s", cid, hf_repo, rejection)
        state.failed_repos.add(hf_repo)
        state.event({"event": "config_rejected", "challenge_id": cid,
                     "hf_repo": hf_repo, "reason": rejection})
        return

    if check_stale:
        current_hash = state.king.get("king_hash", "")
        if not current_hash.startswith(entry["king_hash"][:len(entry["king_hash"])]):
            log.info("stale %s: king changed", cid)
            state.event({"event": "stale", "challenge_id": cid, "hotkey": hotkey})
            return

    block_hash = "default"
    try:
        block_hash = subtensor.get_block_hash(entry["block"]) or "default"
    except Exception:
        pass

    manifest = r2.get("dataset/v1/manifest.json")
    if not manifest:
        log.error("no dataset manifest")
        return
    n_shards = manifest["total_shards"]
    seed_mat = f"{block_hash}:{hotkey}".encode()
    shard_idx = int.from_bytes(hashlib.blake2b(seed_mat, digest_size=8).digest(), "little") % n_shards
    shard_key = manifest["shards"][shard_idx]["key"]

    r2.put(f"eval/{cid}/meta.json", {
        "challenge_id": cid, "king_repo": KING_REPO,
        "challenger_repo": hf_repo, "hotkey": hotkey,
        "N": EVAL_N, "alpha": EVAL_ALPHA, "shard": shard_key,
    })

    state.current_eval = {
        "challenge_id": cid, "challenger_repo": hf_repo, "hotkey": hotkey,
        "progress": 0, "total": EVAL_N, "s": 0, "n": 0,
        "win_rate": 0, "avg_king_loss": 0, "avg_challenger_loss": 0,
        "started_at": _now(),
    }
    state.flush_dashboard()

    verdict = None
    async with httpx.AsyncClient(timeout=httpx.Timeout(1800.0, connect=30.0)) as client:
        resp = await client.post(f"{EVAL_SERVER_URL}/eval", json={
            "king_repo": KING_REPO,
            "challenger_repo": hf_repo,
            "block_hash": block_hash,
            "hotkey": hotkey,
            "shard_key": shard_key,
            "eval_n": EVAL_N,
            "alpha": EVAL_ALPHA,
            "seq_len": SEQ_LEN,
        })
        resp.raise_for_status()
        eval_id = resp.json()["eval_id"]
        log.info("eval %s dispatched to eval server as %s", cid, eval_id)

        async with client.stream("GET", f"{EVAL_SERVER_URL}/eval/{eval_id}/stream",
                                  timeout=httpx.Timeout(1800.0)) as stream:
            async for line in stream.aiter_lines():
                if not line.startswith("data: "):
                    continue
                event = json.loads(line[6:])

                if event["type"] == "progress":
                    d = event["data"]
                    state.current_eval.update({
                        "progress": d.get("done", 0),
                        "total": d.get("total", EVAL_N),
                        "s": d.get("s", 0),
                        "n": d.get("n", 0),
                        "win_rate": d.get("win_rate", 0),
                        "avg_king_loss": d.get("avg_king_loss", 0),
                        "avg_challenger_loss": d.get("avg_challenger_loss", 0),
                    })
                    state.flush_dashboard()

                elif event["type"] == "verdict":
                    verdict = event["data"]
                    verdict["challenge_id"] = cid
                    break

                elif event["type"] == "error":
                    raise RuntimeError(f"eval server error: {event['data']}")

    if not verdict:
        raise RuntimeError("eval stream ended without verdict")

    r2.put(f"eval/{cid}/verdict.json", verdict)
    log.info("verdict: %s (s=%d K=%d wr=%.4f %.1fs)",
             verdict["verdict"], verdict["S_N"], verdict["K"],
             verdict["win_rate"], verdict["wall_time_s"])

    state.current_eval = None
    state.evaluated_repos.add(hf_repo)
    state.record_verdict(verdict, hf_repo, hotkey)

    accepted = verdict.get("accepted", False)
    if accepted:
        state.stats["accepted"] += 1
    else:
        state.stats["rejected"] += 1

    state.flush_dashboard()
    state.event({"event": "eval_completed", "challenge_id": cid,
                 "hotkey": hotkey, "accepted": accepted, **verdict})

    if accepted:
        log.info("DETHRONE! %s wins via %s", hotkey[:16], cid)
        old_hash = state.king.get("king_hash", "")
        new_hash = fork_winner(hf_repo, old_hash, hotkey, cid)
        state.set_king(hotkey, KING_REPO, new_hash, entry.get("block", 0), cid)
        set_weights(subtensor, wallet, NETUID, hotkey)
        state.last_weight_block = subtensor.block

    state.flush()


async def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if not EVAL_SERVER_URL:
        log.error("set TEUTONIC_EVAL_SERVER")
        sys.exit(1)

    r2 = R2()
    state = State(r2)
    state.load()
    if not args.seen:
        state.seen.clear()
        log.info("--no-seen: will continuously re-evaluate all challengers")
    state.flush_dashboard()

    html_path = os.path.join(os.path.dirname(__file__) or ".", "index.html")
    if os.path.exists(html_path):
        with open(html_path, "rb") as f:
            r2.put_raw("index.html", f.read(), "text/html")
        log.info("uploaded dashboard to R2")

    wallet = bt.wallet(name=WALLET_NAME, hotkey=WALLET_HOTKEY)
    subtensor = bt.subtensor(network=NETWORK)

    if not state.king:
        king_hash = "seed"
        try:
            tmp = "/tmp/teutonic/king_seed"
            snapshot_download(KING_REPO, local_dir=tmp, token=HF_TOKEN or None,
                              allow_patterns=["*.safetensors"])
            king_hash = sha256_dir(tmp)
        except Exception:
            pass
        state.set_king(wallet.hotkey.ss58_address, KING_REPO, king_hash, subtensor.block)

    # Verify eval server is reachable
    try:
        r = httpx.get(f"{EVAL_SERVER_URL}/health", timeout=10)
        r.raise_for_status()
        health = r.json()
        log.info("eval server healthy: %s", health)
    except Exception:
        log.warning("eval server at %s not reachable at startup (will retry on eval)", EVAL_SERVER_URL)

    def _on_signal(sig, frame):
        log.info("received signal %d, shutting down", sig)
        sys.exit(0)
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    log.info("validator running | king=%s | eval_server=%s | poll=%ds",
             state.king.get("king_hash", "")[:16], EVAL_SERVER_URL, POLL_INTERVAL)

    while True:
        try:
            state.refresh_metagraph(subtensor, NETUID)
            reveals = scan_reveals(subtensor, NETUID, state.seen)
            if reveals:
                state.flush()
                for rev in reveals:
                    cid = state.enqueue(rev)
                    if cid:
                        log.info("queued %s from %s", cid, rev["hotkey"][:16])

            while state.queue:
                entry = state.queue.pop(0)
                state.flush()
                try:
                    await process_challenge(state, r2, entry, subtensor, wallet,
                                            check_stale=args.seen)
                except Exception:
                    log.exception("eval failed: %s", entry.get("challenge_id"))
                    state.stats["failed"] += 1
                    state.current_eval = None
                    state.flush_dashboard()

            if not args.seen:
                state.seen.clear()
                state.evaluated_repos.clear()
                state.flush()

            try:
                current_block = subtensor.block
                if current_block - state.last_weight_block >= WEIGHT_INTERVAL:
                    king_hotkey = state.king.get("hotkey")
                    if king_hotkey:
                        log.info("periodic weight set at block %d (last=%d)", current_block, state.last_weight_block)
                        set_weights(subtensor, wallet, NETUID, king_hotkey)
                        state.last_weight_block = current_block
            except Exception:
                log.exception("periodic weight-set failed")

        except KeyboardInterrupt:
            break
        except Exception:
            log.exception("tick error")

        await asyncio.sleep(POLL_INTERVAL)


def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seen", action=argparse.BooleanOptionalAction, default=True,
                   help="Track seen hotkeys to avoid re-evaluating (default: True). "
                        "Use --no-seen to continuously cycle all challengers.")
    return p.parse_args()


def main_sync():
    asyncio.run(main())


if __name__ == "__main__":
    main_sync()
