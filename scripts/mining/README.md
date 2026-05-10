# Mining

Train a challenger LLM, beat the current king, win SN3 emission.

This folder is a **two-machine** setup:

```
PC (your wallet, low compute)              GPU box (rented, training only)
   |                                          |
   |- ./submit.sh   <-----  verdict.json --<--|--- ./start.sh   trains, pushes to HF
   |     v                                    |       (writes verdict.json)
   v   SN3 reveal commitment
```

The **GPU box never holds your wallet**. It only has your HF token. Submission is a small, fast tx that runs on your PC where the wallet is safe.

---

## 1. One-time setup on your PC

Create wallet, register hotkey on SN3, note your coldkey prefix.

```bash
btcli wallet new_coldkey --wallet.name teutonic
btcli wallet new_hotkey  --wallet.name teutonic --wallet.hotkey h0
btcli subnet register    --wallet.name teutonic --wallet.hotkey h0 --netuid 3 --network finney

btcli wallet list   # note the COLDKEY ss58 — first 8 chars go in COLDKEY_PREFIX below
```

> **Note on names**: `teutonic` and `h0` are *labels*, not addresses. They name the directory under `~/.bittensor/wallets/teutonic/hotkeys/h0`. The actual ss58 address is computed by bittensor and stored inside `coldkeypub.txt`.

Then in your local clone of the teutonic repo:

```bash
cd scripts/mining
cp .env.example .env
chmod 600 .env
# Edit .env on the PC. You only need:
#   COLDKEY_PREFIX=<first 8 chars of your coldkey ss58>
#   BT_WALLET_NAME=teutonic
#   BT_WALLET_HOTKEY=h0
#   TEUTONIC_NETUID=3
#   TEUTONIC_NETWORK=finney
# HF_TOKEN / WANDB_* / paths / N_GPUS are unused on the PC; leave defaults.
```

## 2. One-time setup on the GPU box

```bash
# clone the repo
git clone https://github.com/unarbos/teutonic
cd teutonic/scripts/mining

# bootstrap (creates venv, installs torch cu128, peft, wandb, etc.)
bash setup.sh
# setup.sh creates .env from the template on first run; rerun after editing.

# Edit .env on the GPU box. You need:
#   HF_TOKEN=<your HF write token>
#   HF_ACCOUNT=<your HF account/org>
#   COLDKEY_PREFIX=<first 8 chars of your coldkey ss58>
#   WANDB_PROJECT=teutonic-mining   (and `wandb login` once)
#   N_GPUS=<your GPU count>
# BT_WALLET_NAME / BT_WALLET_HOTKEY are unused for training-only; leave them.
vim .env

wandb login    # paste W&B API key once
bash setup.sh  # rerun to verify everything imports
```

## 3. Validate the pipeline (do this once, ~20 min, ~$10 of GPU)

```bash
# on the GPU box
./smoke.sh
```

Reads a tiny shard, runs a few SGD steps, merges LoRA, runs a 256-seq paired eval, writes a verdict. **No HF push, no on-chain submission.** Confirms the whole stack is wired up.

If you want to also test on-chain plumbing without burning GPU:

```bash
./noise.sh   # uses miner.py to push a noise-perturbed king + submit reveal
             # validator will reject (expected) but proves chain side works
```

## 4. Real training run

```bash
# on the GPU box
./start.sh           # runs in tmux; survives ssh disconnect
./tail.sh            # attach (Ctrl-b d to detach)
./tail.sh status     # one-shot summary instead
./tail.sh log        # plain `tail -f`
```

`start.sh` will:

1. Discover the live king from the dashboard.
2. Pull the king + 8 dataset shards.
3. Score samples, build curriculum (general/hard/easy), train LoRA.
4. Merge LoRA → standalone candidate.
5. Run an offline paired bootstrap eval (5000 sequences).
6. **If accepted**, push the merged model to `<HF_ACCOUNT>/Teutonic-LXXX-<COLDKEY_PREFIX>-v1`.
7. Write `$WORK_DIR/verdict.json` with `king_hash`, `uploaded_repo`, `challenger_hash`, etc.
8. Iterate (warm-starting from the prior best LoRA) up to 5 times if the first attempt didn't clear the floor.

Targets baked into `start.sh` (matches the chain's fixed `delta=0.0025` floor):
- `mu_hat ≥ 0.012` and `lcb ≥ 0.005` ⇒ early stop, push, you're done.

Watch live on W&B: `iter/mu_hat`, `iter/lcb`, `train/loss`, `paired_eval/iter_N/progress_mu_hat`.

## 5. Submit on-chain — from your PC

After `start.sh` finishes with `best.accepted = true`:

```bash
# from your PC (where the wallet lives)
scp <gpu-box>:/root/teutonic-mining/work/verdict.json ./verdict.json

cd <teutonic-repo>/scripts/mining
./submit.sh ../../verdict.json
# (or pass `--dry-run` to validate without broadcasting)
```

`submit.sh` will:

- Re-validate `best.accepted == true` and that `uploaded_repo` / `challenger_hash` are present.
- Re-check that the HF repo name contains your `COLDKEY_PREFIX`.
- Print the reveal payload `<king_hash[:16]>:<repo>:<chall_hash>` and ask for confirmation.
- Sign with your hotkey, broadcast `set_reveal_commitment` on netuid 3.

After ~30 seconds the validator picks it up. Watch <https://teutonic.ai/dashboard.json> for your verdict.

## 6. Re-evaluate against the LIVE king (optional)

If the king changed between training and submission, re-run the offline test before paying for a tx:

```bash
# on the GPU box, against a saved candidate
./eval.sh /root/teutonic-mining/work/iter_03/merged
```

Writes a fresh `eval-verdict-<ts>.json` next to the merged model. If it still says `accepted: true` against the new king, proceed with `submit.sh` on the PC.

---

## File reference

| File | Where | Purpose |
|---|---|---|
| `setup.sh` | GPU | One-time: venv, deps, dirs, arch sanity. |
| `smoke.sh` | GPU | ~20-min pipeline validation. No HF/chain side effects. |
| `start.sh` | GPU | Real training. Pushes to HF if accepted. tmux-managed. |
| `tail.sh` | GPU | Attach / log / status / stop / pretty-print verdict. |
| `eval.sh` | GPU | Re-run paired eval on a saved merged model vs live king. |
| `noise.sh` | GPU | Cheap chain-plumbing test via `miner.py`. |
| `submit.sh` | PC | Validate verdict + broadcast on-chain reveal. |
| `train_challenger.py` | GPU | The actual training orchestrator. |
| `submit_challenger.py` | PC | The actual on-chain submitter. |
| `report.py` | either | Generate ops report from verdict.json. |
| `_lib.sh` | both | Shared bash helpers (env load, var checks). |
| `.env` | both | Your secrets. Gitignored. |
| `.env.example` | both | Template. Committed. |

## Common verdicts on the dashboard

| `verdict` | `error_code` | What it means | Fix |
|---|---|---|---|
| `challenger` | — | You won. You are king. | nothing — collect SN3 emission |
| `king` | — | Didn't beat (LCB ≤ 0.0025). | retrain harder, more data, bigger LoRA, more epochs |
| `error` | `coldkey_required` | HF repo missing your coldkey ss58 prefix | check `COLDKEY_PREFIX` in `.env` and rename HF repo |
| `error` | `config_mismatch` | challenger config diverges from king | usually `auto_map` snuck in or a `*.py` file shipped — clean repo |
| `error` | `eval_error` | model didn't load on validator | re-merge, ensure no `auto_map`, no `*.py`, only safetensors |
| — | `untrainable:nan` | trainability probe NaN'd | lower noise / LR, check for collapsed projections |

## Troubleshooting

- **`HF_TOKEN looks like a placeholder`** — you forgot to edit `.env`.
- **`UPLOAD_REPO does not contain COLDKEY_PREFIX`** — `start.sh` checks this before launch. Either fix `COLDKEY_PREFIX` (8 chars from `btcli wallet list`) or pass a custom tag: `./start.sh <coldkeyprefix-v2>`.
- **`no tmux session 'teutonic-miner'`** when running `tail.sh` — `start.sh` hasn't been launched yet (or `tail.sh stop` killed it).
- **CUDA OOM during LoRA train** — lower `--lora-r` (the script uses 32; 16 is safe) or switch optimizer (already `paged_adamw_8bit` by default).
- **Inner trainer ImportError on `bitsandbytes`** — `setup.sh` installs it; if you're not using `--optim paged_adamw_8bit`, you can switch to `--optim adamw_torch_fused` and skip bnb.
- **Windows line endings on shells** (`\r: command not found`) — run `dos2unix scripts/mining/*.sh` on the GPU box once.
- **W&B not logging** — either `wandb login` once on the box, or set `WANDB_API_KEY` in `.env`.
- **Verdict's `best.accepted = false`** — read `iter/lcb` and `iter/mu_hat` on W&B. If `mu_hat > 0.005` but `lcb < 0.0025`, you have signal but high variance — bump `--n-eval` (re-run via `./eval.sh` first to see if it certifies on more samples).
