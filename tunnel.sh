#!/usr/bin/env bash
exec ssh -N \
  -L 9000:localhost:9000 \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -o ExitOnForwardFailure=yes \
  -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/dev/null \
  wrk-0638a6gucc7t@ssh.deployments.targon.com
