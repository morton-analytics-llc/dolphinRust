#!/usr/bin/env bash
# Load the working Earthdata credential for real-data validation.
# Safe to commit — holds NO secrets; reads them from ../.env.
# Usage:  source validation/creds.sh   (then run asf_search / earthaccess / dolphin)
#
# Verified 2026-06-17 via asf_search against EDL:
#   * GP_EARTHDATA_TOKEN / EARTHDATA_TOKEN (bearer)  -> auth OK   ← use this
#   * ~/.netrc password for urs.earthdata.nasa.gov   -> 401 STALE ← do NOT use
# Authenticate with the token, e.g.
#   asf_search.ASFSession().auth_with_token(os.environ["GP_EARTHDATA_TOKEN"])
#   earthaccess.login(strategy="environment")   # honours EARTHDATA_TOKEN
set -a
_root="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
[ -f "$_root/.env" ] && . "$_root/.env"
set +a
if [ -n "${GP_EARTHDATA_TOKEN:-}" ]; then
  echo "earthdata: bearer token loaded — authenticate via token (NOT ~/.netrc, which is stale)"
else
  echo "earthdata: TOKEN MISSING — check $_root/.env" >&2
fi
