const { execSync } = require("child_process");

function doppler(key) {
  return execSync(`doppler secrets get ${key} --plain -p arbos -c dev`, { encoding: "utf8" }).trim();
}

function dopplerPrd(key) {
  return execSync(`doppler secrets get ${key} --plain -p arbos -c prd`, { encoding: "utf8" }).trim();
}

module.exports = {
  apps: [{
    name: "teutonic-eval-tunnel",
    script: "./tunnel.sh",
    cwd: "/home/const/workspace",
    autorestart: true,
    restart_delay: 5000,
    max_restarts: 1000,
    log_date_format: "YYYY-MM-DD HH:mm:ss",
  }, {
    name: "teutonic-validator",
    script: "validator.py",
    args: "",
    interpreter: "/home/const/workspace/.venv/bin/python",
    cwd: "/home/const/workspace",
    env: {
      TEUTONIC_EVAL_SERVER: "http://localhost:9000",
      // Active chain (name, seed_repo, repo_pattern, arch) is read from
      // chain.toml at the repo root. Override here only for short-lived
      // experiments — the static file is the source of truth.
      HF_TOKEN: dopplerPrd("HF_TOKEN"),
      HF_HUB_ENABLE_HF_TRANSFER: "1",
      TEUTONIC_NETUID: "3",
      TEUTONIC_NETWORK: "finney",
      BT_WALLET_NAME: "teutonic",
      BT_WALLET_HOTKEY: "default",
      TEUTONIC_R2_ENDPOINT: doppler("R2_URL"),
      TEUTONIC_R2_BUCKET: doppler("R2_BUCKET_NAME"),
      TEUTONIC_R2_ACCESS_KEY: doppler("R2_ACCESS_KEY_ID"),
      TEUTONIC_R2_SECRET_KEY: doppler("R2_SECRET_ACCESS_KEY"),
      TEUTONIC_HIPPIUS_ACCESS_KEY: doppler("HIPPIUS_ACCESS_KEY"),
      TEUTONIC_HIPPIUS_SECRET_KEY: doppler("HIPPIUS_SECRET_KEY"),
      TEUTONIC_DS_ENDPOINT: "https://s3.hippius.com",
      TEUTONIC_DS_BUCKET: "teutonic-sn3",
      TEUTONIC_DS_ACCESS_KEY: doppler("HIPPIUS_ACCESS_KEY"),
      TEUTONIC_DS_SECRET_KEY: doppler("HIPPIUS_SECRET_KEY"),
      TMC_API_KEY: doppler("TMC_API_KEY"),
      DISCORD_BOT_TOKEN: doppler("DISCORD_BOT_TOKEN"),
      DISCORD_CHANNEL_ID: doppler("DISCORD_CHANNEL_ID"),
      // LXXX 80B per-eval wall is ~10 min steady (153 GiB chall HF prefetch
      // ~270s + sharded load ~22s + sharded probe ~4s + bootstrap ~330s at
      // EVAL_N=5000 + cleanup ~15s) and ~14 min cold-page-cache (load grows
      // to ~218s when chall safetensors aren't in OS page cache). The 3600s
      // (1h) restart envelope gives ~4-5x safety margin. Pre-LXXX value was
      // 1800s for the 24B Quasar chain at ~5 min/eval.
      TEUTONIC_TICK_RESTART_AFTER: "3600",
      TEUTONIC_MAX_CONSECUTIVE_TICK_ERRORS: "20",
      // Stream-idle watchdog envelope must accommodate the multi-minute
      // chall HF prefetch + sharded model load. The eval_server emits SSE
      // heartbeat events during these phases so this rarely actually fires
      // in normal operation. Pre-LXXX values were 300/900s.
      TEUTONIC_STREAM_IDLE_WARN_AFTER: "600",
      TEUTONIC_STREAM_IDLE_TIMEOUT: "1800",
      // 165 GiB seed-king HF download + sha256 takes ~25-30 min on this
      // box's network; pre-LXXX default 1200s (20 min) timed out. Bump
      // to 2400s (40 min) for headroom on the next coronation download
      // and any out-of-cycle king-hash recompute (State.load placeholder
      // path).
      TEUTONIC_KING_HASH_TIMEOUT_S: "2400",
    },
    max_restarts: 10,
    restart_delay: 5000,
    autorestart: true,
    log_date_format: "YYYY-MM-DD HH:mm:ss",
  }],
};
