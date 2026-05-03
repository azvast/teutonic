"""Single source of truth for the active king chain.

Reads `chain.toml` at the repo root and exposes constants used by
`validator.py`, `miner.py`, eval/seed/smoke scripts, and the website
(indirectly via dashboard.json). To swap the king to a new generation,
edit `chain.toml` (and add `archs/<new>/` if the architecture changes);
no code edits should be necessary.
"""
from __future__ import annotations

import importlib
import pathlib
import re
import tomllib
from types import ModuleType

_REPO_ROOT = pathlib.Path(__file__).resolve().parent
_TOML_PATH = _REPO_ROOT / "chain.toml"

with open(_TOML_PATH, "rb") as _f:
    _doc = tomllib.load(_f)

_chain = _doc.get("chain", {})
_arch = _doc.get("arch", {})
_seed = _doc.get("seed", {})

NAME: str = _chain["name"]
SEED_REPO: str = _chain["seed_repo"]
REPO_PATTERN: str = _chain.get("repo_pattern") or rf"^[^/]+/{re.escape(NAME)}-.+$"

ARCH_MODULE: str = _arch.get("module", "")
EXTRA_LOCK_KEYS: tuple[str, ...] = tuple(_arch.get("extra_lock_keys", []))

SEED_TOKENIZER_REPO: str = _seed.get("tokenizer_repo", "")

# HF namespace inferred from the seed repo. Miners default their challenger
# repo to "<namespace>/<NAME>-<suffix>" though they can override to publish
# under their own account.
SEED_NAMESPACE: str = SEED_REPO.split("/", 1)[0] if "/" in SEED_REPO else ""


def load_arch() -> ModuleType:
    """Import the configured architecture module.

    The arch package's import side effect is to register its config + model
    classes with HuggingFace `AutoConfig` / `AutoModelForCausalLM` so any
    downstream `from_pretrained` resolves the king without trust_remote_code.
    """
    if not ARCH_MODULE:
        raise RuntimeError("chain.toml is missing [arch].module")
    return importlib.import_module(ARCH_MODULE)


__all__ = [
    "NAME",
    "SEED_REPO",
    "REPO_PATTERN",
    "ARCH_MODULE",
    "EXTRA_LOCK_KEYS",
    "SEED_TOKENIZER_REPO",
    "SEED_NAMESPACE",
    "load_arch",
]
