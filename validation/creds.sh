#!/usr/bin/env bash
# Load Earthdata credentials for real-data validation from the repo .env.
# Safe to commit: holds no secrets and never prints one.
# Usage:  source validation/creds.sh
#
# .env keys (credentials live only there; ~/.netrc is never used):
#   GP_EARTHDATA_TOKEN / EARTHDATA_TOKEN       pasted EDL bearer token (optional)
#   EARTHDATA_USERNAME / EARTHDATA_PASSWORD    used by validation/edl_token.py to
#       get the live token from EDL find_or_create_token when the pasted token
#       is missing, expired or rejected.
# The Python fetch scripts read .env themselves; sourcing this is only needed
# for tools that take the token from the environment (asf_search, earthaccess).
set -a
_root="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
[ -f "$_root/.env" ] && . "$_root/.env"
set +a
if [ -n "${GP_EARTHDATA_TOKEN:-}" ] || [ -n "${EARTHDATA_USERNAME:-}" ]; then
  echo "earthdata: credentials loaded from $_root/.env"
else
  echo "earthdata: no token or EARTHDATA_USERNAME in $_root/.env" >&2
fi
