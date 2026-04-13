#!/usr/bin/env python3
"""Fix config.json rope_parameters for vLLM 0.15.1 compatibility."""
import json
import sys

for model_dir in ["/tmp/teutonic/king", "/tmp/teutonic/challenger"]:
    config_path = f"{model_dir}/config.json"
    with open(config_path) as f:
        cfg = json.load(f)

    if "rope_parameters" in cfg:
        rp = cfg["rope_parameters"]
        if "rope_type" not in rp:
            rp["rope_type"] = "default"
            print(f"Added rope_type to {config_path}")

    if "rope_theta" not in cfg:
        cfg["rope_theta"] = 10000.0
        print(f"Added rope_theta to {config_path}")

    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Fixed {config_path}")
