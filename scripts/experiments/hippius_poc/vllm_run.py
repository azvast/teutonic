"""Stage 4: vLLM cold-load + TTFT + steady-state tok/s on a downloaded model dir.

Falls back to a transformers-only load (no inference) if vLLM refuses the
config (e.g. vendored Quasar arch not registered).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import lib


PROMPT = (
    "You are a thoughtful assistant. Explain in detail how a modern decoder-only "
    "transformer language model generates text token by token, including the role "
    "of attention, KV cache, and sampling temperature."
)


def run_vllm(model_dir: str, tp: int, dtype: str, max_tokens: int, batch: int, log: lib.BenchLogger) -> bool:
    try:
        from vllm import LLM, SamplingParams  # type: ignore
    except Exception as e:
        log.emit("vllm_import_fail", error=str(e))
        return False

    log.emit("vllm_load_start",
             model=model_dir, tensor_parallel_size=tp, dtype=dtype)
    with lib.timer() as t_load:
        try:
            llm = LLM(
                model=model_dir,
                tensor_parallel_size=tp,
                dtype=dtype,
                gpu_memory_utilization=float(os.environ.get("VLLM_GPU_UTIL", "0.9")),
                trust_remote_code=False,
                enforce_eager=False,
            )
        except Exception as e:
            log.emit("vllm_load_fail", error=str(e)[:500])
            return False
    log.emit("vllm_load_done", seconds=round(t_load.elapsed, 3))

    # TTFT: single-prompt, single-token max
    ttft_params = SamplingParams(temperature=0.0, max_tokens=1)
    log.emit("vllm_ttft_start")
    with lib.timer() as t_ttft:
        _ = llm.generate([PROMPT], ttft_params, use_tqdm=False)
    log.emit("vllm_ttft_done", seconds=round(t_ttft.elapsed, 3))

    # Steady state: batch x max_tokens
    sp = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=max_tokens)
    prompts = [PROMPT] * batch
    log.emit("vllm_throughput_start", batch=batch, max_tokens=max_tokens)
    with lib.timer() as t_gen:
        outs = llm.generate(prompts, sp, use_tqdm=False)
    total_completion_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
    tokps = total_completion_tokens / t_gen.elapsed if t_gen.elapsed else 0
    log.emit(
        "vllm_throughput_done",
        seconds=round(t_gen.elapsed, 3),
        completion_tokens=total_completion_tokens,
        tok_per_s=round(tokps, 2),
        per_seq_tok_per_s=round(tokps / max(batch, 1), 2),
    )
    return True


def run_transformers_loadonly(model_dir: str, dtype: str, log: lib.BenchLogger) -> bool:
    try:
        import torch  # type: ignore
        from transformers import AutoConfig, AutoModelForCausalLM  # type: ignore
    except Exception as e:
        log.emit("hf_load_import_fail", error=str(e))
        return False

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(dtype, torch.bfloat16)
    log.emit("hf_load_start", model=model_dir, dtype=dtype)
    with lib.timer() as t:
        cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch_dtype,
            config=cfg,
            trust_remote_code=False,
            device_map="auto",
        )
        # Force at least one tensor onto a GPU to flush load.
        next(model.parameters())
    log.emit("hf_load_done", seconds=round(t.elapsed, 3))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--tensor-parallel-size", type=int, default=8)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--tier", required=True)
    ap.add_argument("--allow-fallback", action="store_true",
                    help="If vLLM fails, fall back to transformers load-only.")
    args = ap.parse_args()

    log = lib.open_logger(args.run_dir, args.tier)

    ok = run_vllm(
        args.model_dir,
        tp=args.tensor_parallel_size,
        dtype=args.dtype,
        max_tokens=args.max_tokens,
        batch=args.batch,
        log=log,
    )
    if not ok:
        if args.allow_fallback:
            log.emit("fallback_transformers", note="vLLM unavailable/incompatible")
            ok2 = run_transformers_loadonly(args.model_dir, args.dtype, log)
            if not ok2:
                log.emit("fallback_failed")
                sys.exit(3)
        else:
            sys.exit(2)


if __name__ == "__main__":
    main()
