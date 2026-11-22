"""Earthdata Login (EDL) bearer token for the validation fetch scripts.

Credentials come only from the process environment or the repo ``.env``:

* ``GP_EARTHDATA_TOKEN`` / ``EARTHDATA_TOKEN``: a pasted user token, used while
  it is present and its JWT ``exp`` is in the future;
* ``EARTHDATA_USERNAME`` / ``EARTHDATA_PASSWORD``: used to get the live token
  from EDL's ``find_or_create_token`` when the pasted token is missing, expired
  or rejected. That endpoint returns the user's live token, or mints one if none
  is live, so it never hits EDL's two-live-tokens cap.

``~/.netrc`` is never read. No token or password is printed or logged.
"""

from __future__ import annotations

import base64
import json
import os
import time
from collections.abc import Callable, Mapping
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
ENV_FILE = ROOT.parent / ".env"
TOKEN_KEYS = ("GP_EARTHDATA_TOKEN", "EARTHDATA_TOKEN")
USERNAME_KEY = "EARTHDATA_USERNAME"
PASSWORD_KEY = "EARTHDATA_PASSWORD"
FIND_OR_CREATE_URL = "https://urs.earthdata.nasa.gov/api/users/find_or_create_token"
# A token this close to expiry is replaced rather than used for a long fetch.
EXPIRY_MARGIN_S = 3600

Mint = Callable[[str, str], str]


def env_file_values(env_file: Path = ENV_FILE) -> dict[str, str]:
    """`KEY=value` pairs from an env file (missing file: none)."""
    if not env_file.is_file():
        return {}
    values = {}
    for line in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
        key, sep, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        if sep and key:
            values[key] = value.strip().strip("'\"")
    return values


def lookup(
    keys: tuple[str, ...],
    environment: Mapping[str, str],
    file_values: Mapping[str, str],
) -> str:
    """First non-empty value of `keys`, environment before env file."""
    for source in (environment, file_values):
        for key in keys:
            value = source.get(key, "").strip()
            if value:
                return value
    return ""


def token_expiry(token: str) -> float | None:
    """The JWT `exp` claim (epoch seconds), or None when it cannot be read."""
    parts = token.split(".")
    if len(parts) != 3:
        return None
    payload = parts[1] + "=" * (-len(parts[1]) % 4)
    try:
        claims = json.loads(base64.urlsafe_b64decode(payload))
    except (ValueError, json.JSONDecodeError):
        return None
    exp = claims.get("exp") if isinstance(claims, dict) else None
    return float(exp) if isinstance(exp, (int, float)) else None


def is_usable(token: str, now: float) -> bool:
    """A present token whose expiry is unknown or comfortably in the future."""
    if not token:
        return False
    expiry = token_expiry(token)
    return expiry is None or expiry - EXPIRY_MARGIN_S > now


def find_or_create_token(username: str, password: str) -> str:
    """The user's live EDL token, minted if none is live."""
    response = requests.post(FIND_OR_CREATE_URL, auth=(username, password), timeout=60)
    if response.status_code != 200:
        raise RuntimeError(
            f"EDL find_or_create_token returned HTTP {response.status_code};"
            f" check {USERNAME_KEY}/{PASSWORD_KEY} in .env"
        )
    token = response.json().get("access_token", "")
    if not token:
        raise RuntimeError("EDL find_or_create_token returned no access_token")
    return token


def earthdata_token(
    environment: Mapping[str, str] = os.environ,
    *,
    env_file: Path = ENV_FILE,
    rejected: bool = False,
    mint: Mint = find_or_create_token,
    now: float | None = None,
) -> str:
    """A usable EDL bearer token.

    The pasted token is returned while usable and not `rejected`; otherwise the
    live token is fetched with the username/password. Fails, naming the keys
    but never a value, when neither works.
    """
    file_values = env_file_values(env_file)
    pasted = lookup(TOKEN_KEYS, environment, file_values)
    if not rejected and is_usable(pasted, time.time() if now is None else now):
        return pasted
    username = lookup((USERNAME_KEY,), environment, file_values)
    password = lookup((PASSWORD_KEY,), environment, file_values)
    if not (username and password):
        state = "was rejected" if rejected and pasted else "is missing or expired"
        raise RuntimeError(
            f"No usable Earthdata token: {'/'.join(TOKEN_KEYS)} {state}, and"
            f" {USERNAME_KEY}/{PASSWORD_KEY} are not set in the environment or .env"
        )
    return mint(username, password)
