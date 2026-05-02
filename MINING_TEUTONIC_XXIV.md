# Mining Teutonic-XXIV (Quasar 8B-active / 24B-total MoE)

The Teutonic SN3 king is now `unconst/Teutonic-XXIV`, a freshly-initialised
SILX-AI Quasar hybrid MoE: ~8B active per token, ~24B total parameters,
RoPE θ=1e6, vocab 262144 (same Hippius shards as before — no retokenization).

This guide tells you how to:

1. Set up your environment.
2. Build / train a challenger.
3. Submit it on-chain.

If you only want to play with random noise, jump to **Quick start (noise
miner)**. If you want to actually dethrone the king, see **Real training**.

---

## 0. The mechanism in one paragraph

The validator pulls every reveal commitment from chain, downloads the
challenger from HF, runs a paired cross-entropy test against the king on a
random Hippius shard, and crowns the challenger if its bootstrap LCB on the
per-token NLL improvement clears `delta = 1/N`. Winner takes 100% of SN3
emission until dethroned. Full mechanism in
[`teutonic/DESIGN.md`](DESIGN.md).

The architecture lock is enforced by `validate_challenger_config` in
[`teutonic/validator.py`](validator.py): your challenger's `config.json`
must match the king on every key in `CONFIG_MATCH_KEYS` (vocab, dims, MoE
shape, looped depth, latent-memory shape, RoPE, …) and **must not** ship
any `*.py` files or set `auto_map`. Vendored modeling code only.

---

## 1. Environment

You need Python 3.12, CUDA 12.8, PyTorch 2.11 (cu128 wheel — earlier wheels
do not have B200 / sm_100 kernels), `transformers >= 5.5`, and our
flash-linear-attention fork that ships the Quasar/GLA layers.

```bash
python3.12 -m venv .venv
. .venv/bin/activate

pip install --index-url https://download.pytorch.org/whl/cu128 torch
pip install transformers accelerate safetensors huggingface_hub bittensor numpy

# The Quasar attention layers + GLA cache live in this fork pinned by SILX:
pip install "flash-linear-attention @ git+https://github.com/SILX-LABS/quasar-flash-linear-attention.git@84ad1cc5a7428609d7e0e56d4041a775cd19b7bb"
```

Clone the validator/miner code so you have the vendored Quasar package
locally:

```bash
git clone https://github.com/unarbos/teutonic
cd teutonic
```

Sanity-check the model loads without `trust_remote_code`:

```bash
python -c "
import sys; sys.path.insert(0, '.')
import teutonic.quasar
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained('unconst/Teutonic-XXIV', torch_dtype='bfloat16', device_map={'': 'cuda:0'})
print('loaded', sum(p.numel() for p in m.parameters())/1e9, 'B params')
"
```

`attn_implementation` will fall back to `eager` for Quasar layers — that's
expected (FA2 / SDPA upstream do not yet support QuasarForCausalLM).

---

## 2. Architecture you must match exactly

The king config is locked at:

| field | value |
|---|---|
| `model_type` | `quasar` |
| `architectures` | `["QuasarForCausalLM"]` |
| `vocab_size` | `262144` (Teutonic-I tokenizer) |
| `d_model` / `hidden_size` | `4096` |
| `n_layers` / `num_hidden_layers` | `32` |
| `n_heads` / `num_attention_heads` | `32` |
| `head_dim` | `128` |
| `d_ff` / `intermediate_size` | `11008` |
| `quasar_layers` / `gated_layers` | `4` / `2` (cycle of 6) |
| `dense_input_layers` | `4` |
| `moe_type` | `bigmac` |
| `num_routed_experts` / `top_k` | `56` / `8` |
| `routed_expert_size` (effective) | `1024` |
| `shared_expert_size` | `2048` |
| `bigmac_r` | `0.25` (DCCA bottleneck) |
| `memory_slots` / `memory_dim` | `128` / `128` |
| `num_loops` | `1` |
| `tie_word_embeddings` | `true` |
| `rope_theta` | `1_000_000` |
| `max_seq_len` / `max_position_embeddings` | `16384` |

If any of these drift, the validator rejects with `"<key> mismatch"`.

If your repo contains `*.py` or your config has `auto_map`, the validator
rejects with `"repo ships *.py files"` or `"auto_map present in
config.json"`. The vendored `teutonic/quasar/` module is the only path the
network accepts — your weights must load via plain
`AutoModelForCausalLM.from_pretrained(...)` after `import teutonic.quasar`.

---

## 3. Repo naming and anti-impersonation

Your HF repo MUST match `^[^/]+/Teutonic-XXIV-.+$` AND embed the first
8 ss58 chars of your coldkey somewhere in the full repo id (case
insensitive, in either the namespace or the model basename). Examples for
coldkey `5DhAqMpdABCDEFG…`:

- ✅ `myaccount/Teutonic-XXIV-5DhAqMpd-v3`
- ✅ `5DhAqMpd/Teutonic-XXIV-noise01`
- ❌ `myaccount/Teutonic-XXIV-v3` (no coldkey prefix)

This is the anti-impersonation gate added 2026-04-29.

---

## 4. Quick start — noise miner

For testing only; will almost never dethrone but verifies your end-to-end
pipeline works.

```bash
. .venv/bin/activate
export HF_TOKEN=hf_...                 # write access to your HF org
export BT_WALLET_NAME=mywallet         # registered on SN3
export BT_WALLET_HOTKEY=default

python teutonic/miner.py \
    --hf-account myaccount \
    --suffix 5DhAqMpd-noise-01 \
    --noise 1e-4
```

Under the hood `miner.py`:

1. Pulls the king at its pinned commit SHA.
2. Adds Gaussian noise of stdev `--noise` to every learnable tensor — but
   skips SMEBU global bias / momentum / max_vio buffers and the latent
   memory state (perturbing those collapses routing or destroys memory).
3. Runs the same `validate_local_config` checks the validator runs.
4. Uploads to `myaccount/Teutonic-XXIV-5DhAqMpd-noise-01`.
5. Submits the on-chain reveal commitment.

You can watch the validator pick it up at
[`https://s3.hippius.com/teutonic-sn3/dashboard.json`](https://s3.hippius.com/teutonic-sn3/dashboard.json).

---

## 5. Real training

You need to lower the king's per-token NLL by more than `delta` nats on a
random unseen Hippius shard, with a one-sided 99.9% bootstrap LCB > delta.
Right now the king is uniform over 262144 tokens (ln(262144) ≈ 12.48), so
the first real training run will dethrone.

A reasonable starting point uses
[`teutonic/scripts/mining/train_challenger.py`](scripts/mining/train_challenger.py)
which:

1. Reads the king repo + revision from the live dashboard.
2. Pulls the king and a few Hippius shards (already tokenized to vocab
   262144).
3. Trains a LoRA adapter (default targets cover Quasar's `q/k/v/o_proj`,
   `ffn.gate/up/down`, `w_down_proj/w_up_proj`).
4. Merges LoRA into the base weights → standalone candidate.
5. Runs an offline paired-CE test against the king to estimate mu_hat
   before burning a HF push and chain reveal.

```bash
torchrun --nproc-per-node=8 teutonic/scripts/mining/train_challenger.py \
    --upload-repo myaccount/Teutonic-XXIV-5DhAqMpd-v1 \
    --noise-only false \
    --max-iters 3
```

Notes specific to Quasar:

- `nn.Parameter` blocks like `experts_w12` and `experts_w3` are NOT Linear
  layers, so PEFT/LoRA cannot target them. Train them with full SGD if you
  want to move the routed experts.
- The SMEBU bias buffer (`model.all_moe_bias`) is updated by the model
  itself only when `model.training=True`. Don't manually overwrite it
  unless you understand what the routing-stability path is doing.
- Latent memory state is reinitialized every forward call when
  `memory_states` is None (default), so you don't have to manage it during
  training — but DO call `model.eval()` before paired evaluation so SMEBU
  doesn't shift bias mid-test.
- `attn_implementation="eager"` is currently the only supported path for
  QuasarForCausalLM under transformers ≤ 5.7. SDPA / FA2 will be wired up
  later.

After training, run the offline paired test the harness emits — if your
estimated `mu_hat` is at least 2-3x the offline `delta`, push and submit.
Otherwise re-run with more steps / different seed / different data
weighting.

---

## 6. What the validator will tell you

Verdicts you might see in `dashboard.json` under `history[*]`:

- `accepted: true, verdict: "challenger"` — you are king.
- `verdict: "king"` — you didn't beat the king (LCB ≤ delta).
- `verdict: "error"` with `error_code: "config_mismatch"` —
  `validate_challenger_config` rejected your repo (read `error_detail`).
- `verdict: "error"` with `error_code: "eval_error"` and
  `"could not load model with any attention implementation"` — your
  safetensors didn't load on the eval server. Most common cause: you
  perturbed SMEBU buffers or shipped a config with `auto_map`.
- `rejection_reason: "untrainable:seed0(...):loss_non_finite:nan"` — the
  trainability probe took one SGD step on your model and got NaN. Means
  your weights are pathological (often: noise too large, or projections
  collapsed). Lower `--noise` or check the reparam-trick guard.

---

## 7. Useful links

- King model: <https://huggingface.co/unconst/Teutonic-XXIV>
- Live dashboard: <https://s3.hippius.com/teutonic-sn3/dashboard.json>
- Source: <https://github.com/unarbos/teutonic>
- Discord: `γ・τeuτonic・3` (ARbos answers technical questions there)
- SILX Quasar docs: <https://huggingface.co/silx-ai/Quasar-3B-A1B-Preview>

---

## 8. FAQ

**Q: Can I just upload my own MoE / dense / Mamba checkpoint?**
A: No. The validator pins `model_type=quasar` and the full Quasar dim set.
Cross-architecture submissions are rejected at config-match.

**Q: Why isn't FlashAttention-2 used?**
A: `QuasarForCausalLM` doesn't yet have an FA2 path in upstream
transformers. The eval server falls back to `eager`. This roughly doubles
forward wall vs FA2 but is otherwise correct.

**Q: Can I train and submit a quantized challenger?**
A: Eval loads in bf16. Quantized weights would dequant on load — usually
fine for storage savings, but be careful about bias drift. `safetensors`
only, no pickle.

**Q: How do I reset my submission if I made a mistake?**
A: You can't dethrone yourself. Wait for the next reign and submit again.
The validator de-dupes per-hotkey within a reign.
