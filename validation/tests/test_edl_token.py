from __future__ import annotations

import base64
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import edl_token


def jwt(exp: int) -> str:
    def part(value: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(value).encode()).decode().rstrip("=")

    return f"{part({'alg': 'RS256'})}.{part({'exp': exp})}.signature"


NOW = 1_800_000_000


class EarthdataTokenTest(unittest.TestCase):
    def setUp(self) -> None:
        self.minted: list[tuple[str, str]] = []

    def mint(self, username: str, password: str) -> str:
        self.minted.append((username, password))
        return "minted-token"

    def token(self, environment: dict[str, str], env_text: str = "", **kwargs) -> str:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text(env_text)
            return edl_token.earthdata_token(
                environment, env_file=env_file, mint=self.mint, now=NOW, **kwargs
            )

    def test_unexpired_pasted_token_is_used_without_minting(self) -> None:
        live = jwt(NOW + 86_400)
        self.assertEqual(self.token({"GP_EARTHDATA_TOKEN": live}), live)
        self.assertEqual(self.minted, [])

    def test_expired_or_missing_token_is_replaced_from_env_file_credentials(self) -> None:
        creds = "EARTHDATA_USERNAME=user\nEARTHDATA_PASSWORD='pw'\n"
        expired = f"GP_EARTHDATA_TOKEN={jwt(NOW - 10)}\n"
        self.assertEqual(self.token({}, expired + creds), "minted-token")
        self.assertEqual(self.token({}, creds), "minted-token")
        self.assertEqual(self.minted, [("user", "pw"), ("user", "pw")])

    def test_rejected_token_is_replaced(self) -> None:
        env = {
            "EARTHDATA_TOKEN": jwt(NOW + 86_400),
            "EARTHDATA_USERNAME": "user",
            "EARTHDATA_PASSWORD": "pw",
        }
        self.assertEqual(self.token(env, rejected=True), "minted-token")

    def test_failure_names_the_keys_but_no_value(self) -> None:
        secret = jwt(NOW - 10)
        with self.assertRaises(RuntimeError) as ctx:
            self.token({"GP_EARTHDATA_TOKEN": secret})
        message = str(ctx.exception)
        self.assertIn("EARTHDATA_USERNAME", message)
        self.assertNotIn(secret, message)


if __name__ == "__main__":
    unittest.main()
