#!/usr/bin/env python3
"""LoRA fine-tune the king on pre-tokenized id sequences.

Invoked under torchrun by `scripts/mining/train_challenger.py`.

Optimizations vs the pre-2026-05 version:
  * Chain-aware default `target_modules`: detects Qwen3-MoE vs Quasar from
    `chain_config.ARCH_MODULE` and picks LoRA targets that actually match
    PEFT's end-of-name rule. The pre-2026 defaults included `gate`/`up`/
    `down` which silently matched NOTHING under Qwen3-MoE (whose FFN is
    `gate_proj`/`up_proj`/`down_proj`), so the prior runs were
    attention-only LoRA on a 7.6 B-active MoE — the experts were frozen.
  * Optimizer: `adamw_torch_fused` by default (B200/H100), or
    `paged_adamw_8bit` to slash optimizer-state VRAM when LoRA targets all
    128 experts.
  * Adam betas (`0.9, 0.95`) — LLM fine-tuning sweet spot vs Adam's
    `0.999` second-moment default.
  * Saner warmup (`0.05`) for short runs (the old 0.02 collapsed into ~5
    warmup steps).
  * `max_grad_norm` exposed.
  * Optional W&B logging (set `WANDB_PROJECT` / pass `--wandb-project`).
  * Resume from a previous adapter via `--resume-adapter` so successive
    iterations of the orchestrator can warm-start instead of training from
    base each time.
  * Best-checkpoint and a `metrics.json` summary are persisted next to the
    adapter so the orchestrator can read final eval/train loss without
    re-parsing the trainer state.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, PeftModel, get_peft_model

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
try:
    import chain_config  # noqa: E402
    chain_config.load_arch()
    _ARCH_MODULE = chain_config.ARCH_MODULE
    _CHAIN_NAME = chain_config.NAME
except Exception:  # noqa: BLE001
    _ARCH_MODULE = ""
    _CHAIN_NAME = "unknown"


# ---------------------------------------------------------------------------
# Per-arch LoRA target presets
# ---------------------------------------------------------------------------
# PEFT matches `target_modules` strings against the END of each module's
# qualified name (i.e. `model.layers.0.self_attn.q_proj` matches `q_proj`).
# Substring matches do NOT work — that was the prior bug.
LORA_TARGET_PRESETS: dict[str, list[str]] = {
    # Qwen3-MoE (Teutonic-LXXX): attention + per-expert FFN. The router
    # `mlp.gate` is intentionally excluded — perturbing it shifts routing
    # which the SMEBU-equivalent in Qwen3-MoE handles less gracefully than
    # we'd like for short fine-tunes. Add it back via --lora-target-modules
    # if you want to experiment.
    "archs.qwen3_moe": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    # Quasar (Teutonic-XXIV, archived). Kept for back-compat if anyone
    # flips chain.toml back. `experts_w12`/`experts_w3` are nn.Parameter,
    # not Linear, so PEFT can't target them.
    "archs.quasar": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "ffn.gate", "ffn.up", "ffn.down",
        "w_down_proj", "w_up_proj",
    ],
}


def default_target_modules() -> list[str]:
    if _ARCH_MODULE in LORA_TARGET_PRESETS:
        return LORA_TARGET_PRESETS[_ARCH_MODULE]
    # Generic LLM fallback — matches HF Llama/Qwen/Mistral-style names.
    return [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class TokenIdsDataset(Dataset):
    def __init__(self, path, seq_len):
        self.rows = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    ids = obj["input_ids"][:seq_len]
                    if len(ids) < seq_len:
                        continue
                    self.rows.append(ids)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        ids = self.rows[idx]
        x = torch.tensor(ids, dtype=torch.long)
        return {
            "input_ids": x,
            "attention_mask": torch.ones_like(x),
            "labels": x.clone(),
        }


@dataclass
class Collator:
    def __call__(self, features):
        return {
            "input_ids": torch.stack([f["input_ids"] for f in features]),
            "attention_mask": torch.stack([f["attention_mask"] for f in features]),
            "labels": torch.stack([f["labels"] for f in features]),
        }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--train-data", required=True)
    ap.add_argument("--val-data", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--micro-batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--warmup-ratio", type=float, default=0.05,
                    help="Fraction of total steps spent in linear warmup. "
                         "Cosine decays from 1.0 -> ~0.0 after warmup.")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--adam-beta1", type=float, default=0.9)
    ap.add_argument("--adam-beta2", type=float, default=0.95,
                    help="LLM fine-tunes converge better with 0.95 than the "
                         "Adam default 0.999.")
    ap.add_argument("--adam-epsilon", type=float, default=1e-8)
    ap.add_argument("--lr-scheduler-type", default="cosine",
                    choices=["linear", "cosine", "cosine_with_restarts",
                             "polynomial", "constant", "constant_with_warmup"])
    ap.add_argument("--optim", default="adamw_torch_fused",
                    choices=["adamw_torch", "adamw_torch_fused",
                             "paged_adamw_8bit", "paged_adamw_32bit",
                             "adafactor"],
                    help="`paged_adamw_8bit` saves ~6x optimizer-state VRAM, "
                         "useful when targeting all Qwen3-MoE experts.")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.0,
                    help="MUST be 0.0 when LoRA targets `nn.Parameter` blocks "
                         "(e.g. Qwen3-MoE fused expert weights). PEFT's "
                         "ParamWrapper raises `does not work with "
                         "lora_dropout != 0.` otherwise. Standard `nn.Linear` "
                         "targets accept any value.")
    ap.add_argument("--lora-target-modules", type=str, default=None,
                    help="comma-separated module name suffixes. Default is "
                         f"chain-aware (current arch={_ARCH_MODULE!r}).")
    ap.add_argument("--lora-rslora", action="store_true",
                    help="Use rank-stabilized LoRA (alpha / sqrt(r) scaling). "
                         "Useful when bumping lora-r beyond ~32.")
    ap.add_argument("--resume-adapter", default="",
                    help="Path to a prior LoRA adapter to warm-start from. "
                         "Lets the orchestrator's iteration N+1 continue "
                         "from iteration N's best instead of re-LoRAing base.")
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--eval-steps", type=int, default=50)
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--save-total-limit", type=int, default=2)
    ap.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", ""))
    ap.add_argument("--wandb-run-name", default=os.environ.get("WANDB_RUN_NAME", ""))
    ap.add_argument("--wandb-tags", default="",
                    help="comma-separated tags forwarded to wandb")
    args = ap.parse_args()

    is_main = (int(os.environ.get("LOCAL_RANK", "0")) == 0)

    # ------------------------------------------------------------------
    # W&B setup (rank-0 only) — must happen BEFORE TrainingArguments so
    # `report_to="wandb"` picks up the env vars.
    # ------------------------------------------------------------------
    use_wandb = bool(args.wandb_project)
    if use_wandb:
        os.environ["WANDB_PROJECT"] = args.wandb_project
        if args.wandb_run_name:
            os.environ["WANDB_RUN_NAME"] = args.wandb_run_name
        if args.wandb_tags:
            os.environ["WANDB_TAGS"] = args.wandb_tags
        os.environ.setdefault("WANDB_LOG_MODEL", "false")
        os.environ.setdefault("WANDB_WATCH", "false")
    report_to = "wandb" if use_wandb else "none"

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        use_safetensors=True,
    )
    model.config.use_cache = False
    # `gradient_checkpointing` requires `use_reentrant=False` for PEFT to
    # propagate gradients through the LoRA adapters cleanly.
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    target_modules = (
        [s.strip() for s in args.lora_target_modules.split(",") if s.strip()]
        if args.lora_target_modules
        else default_target_modules()
    )

    if args.resume_adapter and os.path.isdir(args.resume_adapter):
        if is_main:
            print(f"[lora] resuming from adapter: {args.resume_adapter}",
                  flush=True)
        model = PeftModel.from_pretrained(
            model, args.resume_adapter, is_trainable=True,
        )
    else:
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
            use_rslora=args.lora_rslora,
        )
        model = get_peft_model(model, lora_cfg)

    if is_main:
        model.print_trainable_parameters()
        print(f"[lora] arch={_ARCH_MODULE!r} chain={_CHAIN_NAME!r} "
              f"targets={target_modules}", flush=True)

    train_ds = TokenIdsDataset(args.train_data, args.seq_len)
    val_ds = TokenIdsDataset(args.val_data, args.seq_len)
    if is_main:
        print(f"[lora] train={len(train_ds)} val={len(val_ds)} "
              f"seq_len={args.seq_len}", flush=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.micro_batch_size,
        per_device_eval_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
        lr_scheduler_type=args.lr_scheduler_type,
        optim=args.optim,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=torch.cuda.is_available(),
        fp16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=report_to,
        run_name=args.wandb_run_name or None,
        ddp_find_unused_parameters=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_pin_memory=True,
        # Bumped from 2 → 8: B200 CPUs are 224 cores; the dataloader was
        # CPU-bound on the prior config (tokens are pre-tokenized but still
        # need batching + pin-memory copies).
        dataloader_num_workers=8,
        dataloader_prefetch_factor=4,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=Collator(),
    )

    train_result = trainer.train()
    metrics = dict(train_result.metrics)

    eval_metrics = trainer.evaluate()
    metrics.update({f"final_{k}": v for k, v in eval_metrics.items()})
    try:
        metrics["final_perplexity"] = math.exp(eval_metrics["eval_loss"])
    except (OverflowError, ValueError, KeyError):
        metrics["final_perplexity"] = float("inf")

    adapter_dir = os.path.join(args.output_dir, "best_adapter")
    trainer.save_model(adapter_dir)
    if is_main:
        tokenizer.save_pretrained(adapter_dir)
        with open(os.path.join(adapter_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2, default=float)
        print(f"[lora] saved adapter -> {adapter_dir}", flush=True)
        print(f"[lora] final eval_loss={eval_metrics.get('eval_loss', float('nan')):.6f} "
              f"ppl={metrics['final_perplexity']:.4f}", flush=True)


if __name__ == "__main__":
    main()
