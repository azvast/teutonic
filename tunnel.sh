#!/usr/bin/env bash
# Eval-server SSH tunnel. Targets the 8xB300 SXM6 pod at 95.133.252.44:10310
# (Teutonic-LXXX cutover, 2026-05-07). Each B300 has 275 GiB HBM (vs B200 180 GiB),
# enough to comfortably shard the 153 GiB bf16 LXXX king across 4 GPUs while
# leaving room for the challenger replica on the other 4.
# Previous: Lium 8xB200 pod `teutonic-eval` at 95.133.252.200:10100 (Quasar 24B chain).
exec ssh -N \
  -L 9000:localhost:9000 \
  -p 10310 \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -o ExitOnForwardFailure=yes \
  -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/dev/null \
  root@95.133.252.44
