#!/usr/bin/env python3
"""Splice a lost reign back into the rolling king chain on R2.

Background
----------
On 2026-05-06 ~03:06 UTC, eval-0312 (scoutminer/Teutonic-XXIV-5DDCmgAV-1700,
hotkey 5C4mBwvta7adoDQQ..., UID 114) was accepted as the new king. The
verdict was recorded into dashboard_history.json and the validator briefly
set weights crediting that hotkey at 20%. But the corresponding set_king()
chain mutation never durably flushed to R2 (the process was bounced before
the next state.flush()), and on the subsequent restart the chain-reconcile
sanity check tripped:

    chain reconcile: history top hk=5C4mBwvta7adoDQQ != king hk=5DNtQADRzLt5BXD4;
    skipping reconcile

So the next dethrone (eval-0316, scoutminer-1800) ran against the *old*
king (iter03) and replaced state.king without -1700 ever appearing in the
linked list. From that point on, -1700 sits in dashboard_history with
accepted=True but is invisible to aggregate_chain_weights() and so earns
nothing.

This script surgically inserts the missing reign back between eval-0316
(slot 1 in the current chain) and eval-0305 (slot 2), and shifts every
reign_number above the insertion point up by 1.

Run this with the validator stopped:

    pm2 stop teutonic-validator
    env $(doppler secrets download --no-file --format env -p arbos -c dev \\
        | grep -E '^R2_|^HF_TOKEN' | xargs) \\
        python3 scripts/restore_lost_reign.py            # dry-run preview
    env ... python3 scripts/restore_lost_reign.py --apply
    pm2 start teutonic-validator
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from datetime import datetime, timezone

import boto3
from botocore.config import Config as BotoConfig
from huggingface_hub import HfApi


LOST_CID = "eval-0312"
LOST_REPO = "scoutminer/Teutonic-XXIV-5DDCmgAV-1700"
LOST_HOTKEY = "5C4mBwvta7adoDQQMTdwUW8oRdBbvuxQVkzUbzTtUfXwGEf2"
LOST_REV_PREFIX = "2f372460a37f"
LOST_CROWNED_AT = "2026-05-06T03:06:07.413885+00:00"
INSERT_AFTER_CID = "eval-0316"
INSERT_BEFORE_CID = "eval-0305"


def r2_client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_URL"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
        config=BotoConfig(connect_timeout=15, read_timeout=45,
                          retries={"max_attempts": 3, "mode": "adaptive"}),
    )


def r2_get_json(c, bucket, key):
    return json.loads(c.get_object(Bucket=bucket, Key=key)["Body"].read())


def r2_put_json(c, bucket, key, data):
    c.put_object(Bucket=bucket, Key=key,
                 Body=json.dumps(data, default=str).encode(),
                 ContentType="application/json")


def resolve_full_revision(repo: str, prefix: str) -> str:
    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    for c in api.list_repo_commits(repo):
        if c.commit_id.startswith(prefix):
            return c.commit_id
    raise SystemExit(f"could not find commit starting with {prefix} in {repo}")


def print_chain(label: str, king: dict) -> None:
    print(f"=== {label} ===")
    node = king
    i = 0
    while node:
        print(
            f"  [{i}] cid={node.get('challenge_id')} reign={node.get('reign_number')} "
            f"repo={node.get('hf_repo')} hk={(node.get('hotkey') or '')[:16]}"
        )
        node = node.get("previous_king")
        i += 1


def splice(king: dict, lost_node: dict, after_cid: str) -> dict:
    """Return a deep copy of `king` with `lost_node` inserted right after the
    chain entry whose challenge_id == after_cid. Reign numbers from the head
    down to (but not including) the insertion point are bumped by +1; the
    inserted node's reign_number is set to (anchor.reign_number - 1) so the
    chain stays strictly decreasing.
    """
    head = copy.deepcopy(king)
    cur = head
    while cur is not None:
        cur["reign_number"] = int(cur.get("reign_number") or 0) + 1
        if cur.get("challenge_id") == after_cid:
            tail = cur.get("previous_king")
            new_node = copy.deepcopy(lost_node)
            new_node["reign_number"] = int(cur["reign_number"]) - 1
            new_node["previous_king"] = tail
            cur["previous_king"] = new_node
            return head
        cur = cur.get("previous_king")
    raise SystemExit(f"no chain node with challenge_id={after_cid} found")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="Actually write to R2. Default is dry-run.")
    args = ap.parse_args()

    bucket = os.environ["R2_BUCKET_NAME"]
    c = r2_client()

    state = r2_get_json(c, bucket, "state/validator_state.json")
    king_current = r2_get_json(c, bucket, "king/current.json")
    history_blob = r2_get_json(c, bucket, "state/dashboard_history.json")
    history = history_blob.get("history", [])

    if state["king"].get("challenge_id") != king_current.get("challenge_id"):
        print("[warn] state.king and king/current.json are out of sync; "
              "will overwrite king/current.json from the patched state.king")

    print_chain("BEFORE  state.king linked list", state["king"])

    cids_in_chain = []
    node = state["king"]
    while node:
        cids_in_chain.append(node.get("challenge_id"))
        node = node.get("previous_king")
    if LOST_CID in cids_in_chain:
        print(f"\n[abort] {LOST_CID} is already in the chain — nothing to do")
        return
    if INSERT_AFTER_CID not in cids_in_chain:
        print(f"\n[abort] insertion anchor {INSERT_AFTER_CID} not present in chain "
              f"(chain has {cids_in_chain})")
        return

    lost_history_row = next(
        (h for h in history if h.get("challenge_id") == LOST_CID and h.get("accepted")),
        None,
    )
    if not lost_history_row:
        raise SystemExit(f"no accepted history row found for {LOST_CID}")
    if lost_history_row.get("hotkey") != LOST_HOTKEY:
        raise SystemExit(
            f"history row hotkey {lost_history_row.get('hotkey')!r} != expected {LOST_HOTKEY!r}"
        )

    print(f"\n[hf] resolving full revision for {LOST_REPO} prefix={LOST_REV_PREFIX} ...")
    full_rev = resolve_full_revision(LOST_REPO, LOST_REV_PREFIX)
    print(f"[hf] -> {full_rev}")

    insert_after_node = next(
        n for n in iter_chain(state["king"]) if n.get("challenge_id") == INSERT_AFTER_CID
    )
    insert_after_block = int(insert_after_node.get("crowned_block") or 0)

    lost_node = {
        "hotkey": LOST_HOTKEY,
        "hf_repo": LOST_REPO,
        "king_hash": "dethrone",
        "king_revision": full_rev,
        "reign_number": 0,
        "crowned_at": LOST_CROWNED_AT,
        "crowned_block": max(0, insert_after_block - 1),
        "challenge_id": LOST_CID,
        "previous_king": None,
    }

    new_king = splice(state["king"], lost_node, INSERT_AFTER_CID)
    new_king_current = copy.deepcopy(new_king)
    new_king_current["previous_king"] = None

    print_chain("AFTER   state.king linked list", new_king)

    if not args.apply:
        print("\n[dry-run] not writing R2. re-run with --apply to commit.")
        return

    state["king"] = new_king
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    r2_put_json(c, bucket, "state/validator_state.json", state)
    print("[r2] wrote state/validator_state.json")

    r2_put_json(c, bucket, "king/current.json", new_king_current)
    print("[r2] wrote king/current.json")

    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": "lost_reign_restored",
        "challenge_id": LOST_CID,
        "hotkey": LOST_HOTKEY,
        "hf_repo": LOST_REPO,
        "king_revision": full_rev,
        "reason": (
            "eval-0312 was accepted in dashboard_history but never persisted "
            "to state.king (set_king flush dropped); chain-reconcile sanity "
            "check on subsequent restart bailed out, leaving the reign lost"
        ),
        "trigger": "operator_script:restore_lost_reign",
    }
    try:
        existing = c.get_object(Bucket=bucket, Key="state/history.jsonl")["Body"].read()
    except Exception:
        existing = b""
    body = existing + (json.dumps(event) + "\n").encode()
    c.put_object(Bucket=bucket, Key="state/history.jsonl",
                 Body=body, ContentType="application/x-ndjson")
    print("[r2] appended event to state/history.jsonl")

    print("\nDONE. Restart the validator to pick up the patched chain:")
    print("    pm2 start teutonic-validator")


def iter_chain(king):
    node = king
    while node is not None:
        yield node
        node = node.get("previous_king")


if __name__ == "__main__":
    main()
