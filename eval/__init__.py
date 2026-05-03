"""Teutonic eval runners.

Submodules:
- ``torch_runner``: multi-GPU PyTorch paired-bootstrap CE evaluator.
- ``vllm_runner``: drop-in vLLM-backed paired CE evaluator.
- ``vllm_server``: FastAPI eval server using vLLM (alternative to the
  ``eval_server`` at the repo root, not yet wired into production).
"""
