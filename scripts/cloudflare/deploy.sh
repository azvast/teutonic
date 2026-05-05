#!/usr/bin/env bash
# Deploy the teutonic-proxy Cloudflare Worker and purge the edge cache.
#
# Reads CLOUDFLARE_TEUTONIC_TOKEN (a zone+account-scoped API token with
# Workers Scripts:Edit and Cache Purge permissions) from Doppler.
#
# Usage:  scripts/cloudflare/deploy.sh
set -euo pipefail

ACCOUNT_ID="00523074f51300584834607253cae0fa"
ZONE_ID="1075a976f65a8acdfeb5109615bb5906"
WORKER_NAME="teutonic-proxy"

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
worker_path="$repo_root/scripts/cloudflare/teutonic-proxy/worker.mjs"

if [[ ! -f "$worker_path" ]]; then
  echo "worker source not found at $worker_path" >&2
  exit 1
fi

if ! command -v doppler >/dev/null 2>&1; then
  echo "doppler is not installed; install it or export CLOUDFLARE_TEUTONIC_TOKEN manually" >&2
  exit 1
fi

CFTK="$(doppler secrets get CLOUDFLARE_TEUTONIC_TOKEN --plain)"
if [[ -z "$CFTK" ]]; then
  echo "CLOUDFLARE_TEUTONIC_TOKEN is empty in doppler" >&2
  exit 1
fi

api="https://api.cloudflare.com/client/v4"

echo "==> uploading worker $WORKER_NAME"
upload_resp="$(curl -sS -X PUT \
  -H "Authorization: Bearer $CFTK" \
  -F 'metadata={"main_module":"worker.mjs","compatibility_date":"2024-09-23"};type=application/json' \
  -F "worker.mjs=@${worker_path};type=application/javascript+module" \
  "$api/accounts/$ACCOUNT_ID/workers/scripts/$WORKER_NAME")"

if ! echo "$upload_resp" | python3 -c "import json,sys; sys.exit(0 if json.load(sys.stdin).get('success') else 1)"; then
  echo "worker upload failed:" >&2
  echo "$upload_resp" >&2
  exit 1
fi
echo "    ok ($(echo "$upload_resp" | python3 -c 'import json,sys; print(json.load(sys.stdin)["result"]["etag"])'))"

echo "==> purging cloudflare edge cache for zone $ZONE_ID"
purge_resp="$(curl -sS -X POST \
  -H "Authorization: Bearer $CFTK" \
  -H "Content-Type: application/json" \
  --data '{"purge_everything":true}' \
  "$api/zones/$ZONE_ID/purge_cache")"

if ! echo "$purge_resp" | python3 -c "import json,sys; sys.exit(0 if json.load(sys.stdin).get('success') else 1)"; then
  echo "cache purge failed:" >&2
  echo "$purge_resp" >&2
  exit 1
fi
echo "    ok"

echo "==> verifying teutonic.ai headers"
sleep 2
curl -sS -D - -o /dev/null "https://teutonic.ai/?cb=$(date +%s%N)" \
  | grep -iE 'HTTP/|cache-control|last-modified|etag|x-served-by|x-amz-version' \
  || true
