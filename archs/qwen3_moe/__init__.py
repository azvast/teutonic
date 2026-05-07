"""Qwen3-MoE arch shim for Teutonic-LXXX.

Vanilla `Qwen3MoeForCausalLM` ships in `transformers >= 4.51` and is already
self-registered with `AutoConfig` / `AutoModelForCausalLM`, so the import
side-effect of this package is enough to make `chain_config.load_arch()`
work for the LXXX chain. We deliberately do NOT vendor any modeling code
(unlike `archs/quasar/`) — the whole point of switching to Qwen3MoE is to
delete custom modeling and rely on the in-tree implementation.

`size.py` builds the config under `init_empty_weights` and reports
total/active param counts. `seed.py` builds and pushes a freshly initialised
80B Qwen3MoE checkpoint.
"""
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM  # noqa: F401

__all__ = ["Qwen3MoeConfig", "Qwen3MoeForCausalLM"]
