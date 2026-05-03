"""Architecture registry for the Teutonic king-of-the-hill subnet.

Each subdirectory ships a vendored model architecture (config + modeling
code) that registers itself with HuggingFace `AutoConfig` /
`AutoModelForCausalLM` on import. The active arch is selected by
`chain.toml -> [arch].module` and loaded via `chain_config.load_arch()`.

Adding a new arch: drop it under `archs/<name>/` with an `__init__.py`
that calls the same `AutoConfig.register(...)` / `AutoModel.register(...)`
sequence, then point `chain.toml` at it.
"""
