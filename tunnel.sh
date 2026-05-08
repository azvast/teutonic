#!/usr/bin/env bash
# Eval-server SSH tunnel. Targets the 8xB200 pod at 95.133.252.33:10299
# (Teutonic-LXXX migrated 2026-05-08 after the previous B300 pod at .44:10310
# went offline). Each B200 has 180 GiB HBM — comfortable for the 153 GiB bf16
# LXXX king sharded across 4 GPUs (~38 GiB/GPU weights + room for activations).
# Per-GPU shard budget tuned to 120 GiB (vs B300's 240) for activation headroom.
# Previous pods: B300 95.133.252.44:10310 (LXXX cutover); Lium B200 95.133.252.200:10100 (Quasar 24B chain).
exec ssh -N \
  -L 9000:localhost:9000 \
  -p 10299 \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -o ExitOnForwardFailure=yes \
  -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/dev/null \
  root@95.133.252.33
