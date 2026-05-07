#!/usr/bin/env python3
"""Qwen3-MoE config sizer for Teutonic-LXXX. Builds Qwen3MoeConfig +
Qwen3MoeForCausalLM on the meta device and reports total / active params.

Mirrors `archs/quasar/size.py` so the iterate-until-the-numbers-look-right
workflow is identical across arches.

Usage:
    source /home/const/workspace/.venv/bin/activate
    python -m archs.qwen3_moe.size \
        --hidden 4096 --n-layers 36 \
        --num-experts 128 --top-k 8 --moe-intermediate-size 1408
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from accelerate import init_empty_weights

from archs.qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM


def build_config(args) -> Qwen3MoeConfig:
    return Qwen3MoeConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden,
        num_hidden_layers=args.n_layers,
        num_attention_heads=args.n_heads,
        num_key_value_heads=args.n_kv_heads,
        head_dim=args.head_dim,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=args.max_seq_len,
        tie_word_embeddings=args.tie_word_embeddings,
        num_experts=args.num_experts,
        num_experts_per_tok=args.top_k,
        moe_intermediate_size=args.moe_intermediate_size,
        decoder_sparse_step=args.decoder_sparse_step,
        norm_topk_prob=args.norm_topk_prob,
        output_router_logits=False,
        router_aux_loss_coef=args.router_aux_loss_coef,
        mlp_only_layers=args.mlp_only_layers or [],
        rope_theta=args.rope_theta,
        bos_token_id=args.bos_token_id,
        eos_token_id=args.eos_token_id,
        pad_token_id=args.pad_token_id,
    )


def _classify(name: str) -> str:
    n = name.lower()
    if "embed_tokens" in n:
        return "embed"
    if "lm_head" in n:
        return "lm_head"
    # Qwen3MoE stores experts as a ModuleList: model.layers.{l}.mlp.experts.{e}.{gate,up,down}_proj
    # The router is `model.layers.{l}.mlp.gate.weight`.
    if ".experts." in n:
        return "moe_experts"
    if ".mlp.gate.weight" in n and ".experts." not in n:
        return "moe_router"
    if any(k in n for k in (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "q_norm", "k_norm", "self_attn",
    )):
        return "attn"
    if any(k in n for k in ("gate_proj", "up_proj", "down_proj")):
        # mlp_only_layers (dense FFN) or any non-expert MLP
        return "mlp_dense"
    if "norm" in n:
        return "norm"
    return "other"


def count_params(model, cfg: Qwen3MoeConfig):
    """Walk the meta model. For per-expert tensors (under .experts.{i}.) we
    credit (top_k / num_experts) of the param count to the active total —
    each expert appears once in named_parameters() because Qwen3MoE uses a
    ModuleList of expert MLPs (NOT a single fused (E, ...) parameter like
    Quasar's BigMac)."""
    total = 0
    active = 0
    by_class: dict[str, int] = {}
    by_class_active: dict[str, int] = {}

    embed_counted_once = False
    tied = bool(cfg.tie_word_embeddings)

    for name, p in model.named_parameters():
        n = p.numel()

        if "lm_head" in name and tied:
            continue
        if "embed_tokens" in name and tied and embed_counted_once:
            continue
        if "embed_tokens" in name:
            embed_counted_once = True

        bucket = _classify(name)
        by_class[bucket] = by_class.get(bucket, 0) + n
        total += n

        if ".experts." in name:
            # one expert tensor; full routed = num_experts of these; active = top_k of these
            active_n = int(n * cfg.num_experts_per_tok / cfg.num_experts)
        else:
            active_n = n
        active += active_n
        by_class_active[bucket] = by_class_active.get(bucket, 0) + active_n

    return total, active, by_class, by_class_active


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vocab-size", type=int, default=151936)
    p.add_argument("--hidden", type=int, default=4096)
    p.add_argument("--n-layers", type=int, default=36)
    p.add_argument("--n-heads", type=int, default=32)
    p.add_argument("--n-kv-heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--intermediate-size", type=int, default=11008)
    p.add_argument("--max-seq-len", type=int, default=16384)
    p.add_argument("--tie-word-embeddings", action="store_true", default=True)
    p.add_argument("--no-tie", dest="tie_word_embeddings", action="store_false")
    p.add_argument("--num-experts", type=int, default=128)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--moe-intermediate-size", type=int, default=1408)
    p.add_argument("--decoder-sparse-step", type=int, default=1,
                   help="1 means every layer is MoE; >1 alternates dense/MoE")
    p.add_argument("--norm-topk-prob", action="store_true", default=True)
    p.add_argument("--router-aux-loss-coef", type=float, default=0.001)
    p.add_argument("--mlp-only-layers", type=int, nargs="*", default=None,
                   help="layer indices that stay dense (no MoE). Default: none.")
    p.add_argument("--rope-theta", type=float, default=1_000_000.0)
    p.add_argument("--bos-token-id", type=int, default=151643)
    p.add_argument("--eos-token-id", type=int, default=151645)
    p.add_argument("--pad-token-id", type=int, default=151643)
    args = p.parse_args()

    cfg = build_config(args)

    print("config:")
    for k in ("vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads",
              "num_key_value_heads", "head_dim", "intermediate_size",
              "max_position_embeddings", "tie_word_embeddings",
              "num_experts", "num_experts_per_tok", "moe_intermediate_size",
              "decoder_sparse_step", "norm_topk_prob", "router_aux_loss_coef",
              "mlp_only_layers", "rope_parameters"):
        # transformers 5.5+ flattens rope into rope_parameters (a dict);
        # older versions exposed rope_theta directly. Print whichever exists.
        val = getattr(cfg, k, None)
        if val is None and k == "rope_parameters":
            val = getattr(cfg, "rope_theta", None)
        print(f"  {k} = {val}")

    with init_empty_weights():
        model = Qwen3MoeForCausalLM(cfg)

    total, active, by_class, by_class_active = count_params(model, cfg)

    print("\nparams (total / active per token):")
    for bucket in sorted(set(by_class) | set(by_class_active)):
        t = by_class.get(bucket, 0)
        a = by_class_active.get(bucket, 0)
        print(f"  {bucket:14s}  total {t/1e9:7.3f}B   active {a/1e9:7.3f}B")
    print(f"  {'-'*14}  -----------------")
    print(f"  {'TOTAL':14s}  total {total/1e9:7.3f}B   active {active/1e9:7.3f}B")

    bf16_total_gb = total * 2 / (1024 ** 3)
    bf16_active_gb = active * 2 / (1024 ** 3)
    print(f"\nbf16 weight sizes: total {bf16_total_gb:.1f} GiB / active {bf16_active_gb:.1f} GiB per copy")
    print(f"sharded across 4 GPUs: ~{bf16_total_gb/4:.1f} GiB / GPU just for weights")


if __name__ == "__main__":
    main()
