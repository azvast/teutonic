#!/usr/bin/env python3
"""Create synthetic eval dataset and verify model config."""
import json
import numpy as np

vocab_size = 262144
seq_len = 2048
n_sequences = 200

rng = np.random.default_rng(42)
tokens = rng.integers(0, vocab_size, size=(n_sequences * seq_len,), dtype=np.uint32)
np.save("/tmp/teutonic/synthetic_eval.npy", tokens)
print(f"Created: {n_sequences} seqs x {seq_len} tokens, {tokens.nbytes / 1e6:.1f} MB")

with open("/tmp/teutonic/king/config.json") as f:
    cfg = json.load(f)
arch = cfg.get("architectures", ["unknown"])[0]
hs = cfg.get("hidden_size")
nl = cfg.get("num_hidden_layers")
vs = cfg.get("vocab_size")
print(f"Arch: {arch}, hidden={hs}, layers={nl}, vocab={vs}")
