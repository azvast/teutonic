# Teutonic

Holy Pretraining Incentives — Bittensor SN3 (`netuid 3`).

Website: <https://teutonic.ai>

Teutonic is a king-of-the-hill pretraining subnet. Miners train a challenger
LLM and submit it on-chain. A validator pulls the challenger from HuggingFace,
runs a paired-bootstrap cross-entropy duel against the reigning king on a
held-out tokenized stream, and either accepts (challenger becomes the new king)
or rejects. Winner takes 100% of SN3 emission until dethroned.

The current king is `unconst/Teutonic-XXIV` — a freshly-initialised SILX-AI
Quasar hybrid MoE (~8B active, ~24B total). See [`docs/MINING_TEUTONIC_XXIV.md`](docs/MINING_TEUTONIC_XXIV.md)
for the live mining recipe. The full mechanism is documented in
[`docs/DESIGN.md`](docs/DESIGN.md).

## Repo layout

| Path | What it is |
| --- | --- |
| [`validator.py`](validator.py) | Single-file king-of-the-hill validator. Polls chain, dispatches duels to the eval server, manages king lifecycle on HF, persists state to R2. |
| [`miner.py`](miner.py) | Reference miner: clones the king, perturbs weights, uploads to HF, commits the reveal on-chain. |
| [`eval_server.py`](eval_server.py) | Persistent FastAPI service wrapping the eval pipeline. Caches the king across duels. SSE-streams progress to the validator. |
| [`eval/`](eval/) | Eval runners: [`torch_runner.py`](eval/torch_runner.py) (multi-GPU PyTorch paired-bootstrap CE), [`vllm_runner.py`](eval/vllm_runner.py) (vLLM evaluator), [`vllm_server.py`](eval/vllm_server.py) (vLLM-backed alternative eval server, not yet in production). |
| [`quasar/`](quasar/) | Vendored Quasar architecture (config + modeling). Self-registers with HF Auto* on import so checkpoints load without `trust_remote_code`. |
| [`scripts/`](scripts/) | Operator + miner tooling: bot, dashboard, mining harness, dataset reshard, Cloudflare publish, current-chain seed/smoke. |
| [`docs/`](docs/) | Design doc, scoring plan, current-chain mining guide. |
| [`benchmarks/`](benchmarks/) | Captured benchmark snapshots (e.g. ifeval). |
| [`index.html`](index.html), [`manifest.json`](manifest.json), `favicon.*` | Public dashboard assets. The validator uploads `index.html` to Hippius on every restart. |
| [`ecosystem.config.js`](ecosystem.config.js) | PM2 process manifest for the eval-tunnel + validator. Reads secrets via Doppler. |
| [`tunnel.sh`](tunnel.sh) | SSH port-forward to the GPU box hosting the eval server. |

## Setup

Python 3.11+. We use [`uv`](https://github.com/astral-sh/uv) for everything.

```bash
uv venv
source .venv/bin/activate
uv pip install -e .            # base
uv pip install -e ".[dev]"     # base + ruff
```

Secrets are read at runtime from Doppler (project `arbos`). `ecosystem.config.js`
shows every variable the validator expects.

## Running

### Validator + tunnel (production)

PM2 manages both the SSH tunnel to the GPU box and the validator loop:

```bash
pm2 start ecosystem.config.js
pm2 logs teutonic-validator
```

### Eval server (GPU box)

On the GPU machine that the tunnel forwards to:

```bash
uvicorn eval_server:app --host 127.0.0.1 --port 9000
```

`eval_server.py` will lazy-load the king from HF, then sit on it across duels.
HF cache lives under `~/.cache/huggingface/hub/` and is watermark-cleaned by
the server itself.

### Mining

Don't follow the README — it's deliberately thin so it can't go stale. Read
the live recipe at <https://teutonic.ai/mining> (also at
[`docs/MINING_TEUTONIC_XXIV.md`](docs/MINING_TEUTONIC_XXIV.md)) and use
[`scripts/mining/`](scripts/mining/) as a working harness.

## Docs

- [`docs/DESIGN.md`](docs/DESIGN.md) — full mechanism: how the duel scoring,
  config-lock, trainability probe, dethrone rule, and emission incentive fit
  together.
- [`docs/SCORING_PLAN.md`](docs/SCORING_PLAN.md) — exponential dethrone
  scoring rollout plan.
- [`docs/MINING_TEUTONIC_XXIV.md`](docs/MINING_TEUTONIC_XXIV.md) — current
  chain's mining contract and step-by-step recipe.

## License

MIT.
