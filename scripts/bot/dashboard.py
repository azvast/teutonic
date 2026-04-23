#!/usr/bin/env python3
"""Fetch and format the public Teutonic dashboard."""
import json
import sys
import urllib.request

DASHBOARD_URL = "https://s3.hippius.com/teutonic-sn3/dashboard.json"

try:
    with urllib.request.urlopen(DASHBOARD_URL, timeout=10) as resp:
        data = json.loads(resp.read())

    king = data.get("king", {})
    stats = data.get("stats", {})
    queue = data.get("queue", [])
    history = data.get("history", [])[:5]
    market = data.get("market", {})
    current = data.get("current_eval")

    score_window = data.get("score_window", {})
    out = {
        "updated_at": data.get("updated_at"),
        "king": {
            "repo": king.get("hf_repo"),
            "hotkey": king.get("hotkey", "")[:16] + "...",
            "reign": king.get("reign_number"),
            "crowned_at": king.get("crowned_at"),
        },
        "stats": stats,
        "score_window": {
            "window_id": score_window.get("window_id"),
            "started_at": score_window.get("started_at"),
            "started_block": score_window.get("started_block"),
            "top3": [
                {
                    "hotkey": e.get("hotkey", "")[:16] + "...",
                    "repo": e.get("challenger_repo"),
                    "mu_hat": e.get("mu_hat"),
                    "lcb": e.get("lcb"),
                    "challenge_id": e.get("challenge_id"),
                }
                for e in score_window.get("topk", [])[:3]
            ],
            "last_weight_set": score_window.get("last_weight_set"),
        },
        "queue_length": len(queue),
        "queue": [
            {"id": e.get("challenge_id"), "repo": e.get("hf_repo"),
             "uid": e.get("uid"), "reeval": e.get("reeval", False)}
            for e in queue[:5]
        ],
        "recent_evals": [
            {"id": e.get("challenge_id"), "repo": e.get("challenger_repo"),
             "accepted": e.get("accepted"), "verdict": e.get("verdict"),
             "king_loss": e.get("avg_king_loss"), "challenger_loss": e.get("avg_challenger_loss"),
             "timestamp": e.get("timestamp")}
            for e in history
        ],
    }
    if current:
        out["current_eval"] = {
            "id": current.get("challenge_id"),
            "repo": current.get("challenger_repo"),
            "progress": f"{current.get('progress', 0)}/{current.get('total', 0)}",
            "mu_hat": current.get("mu_hat"),
        }
    if market:
        out["market"] = {
            "tao_usd": market.get("tao_price_usd"),
            "alpha_usd": market.get("sn3_alpha_price_usd"),
            "reg_burn_tao": market.get("sn3_reg_burn_tao"),
        }

    print(json.dumps(out, indent=2))
except Exception as e:
    print(json.dumps({"error": str(e)}))
