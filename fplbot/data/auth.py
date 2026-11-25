"""FPL auth via PingOne OIDC refresh token.

FPL retired the old users.premierleague.com email/password login. Authenticated calls now
send `X-API-Authorization: Bearer <access_token>`. We keep a long-lived refresh token in .env,
exchange it for short-lived access tokens, and persist the *rotated* refresh token each time.

One-time bootstrap: copy refresh_token, token URL and client_id from your browser session
(see .env.example). Verify the exact token-request form fields in DevTools on first run —
this is an undocumented flow and FPL can change it.
"""
from __future__ import annotations

import os
import time

import httpx
from dotenv import set_key

from fplbot.config import ENV_FILE


class OIDCAuth:
    def __init__(self):
        self.token_url = os.environ.get("FPL_TOKEN_URL", "")
        self.client_id = os.environ.get("FPL_CLIENT_ID", "")
        self.refresh_token = os.environ.get("FPL_REFRESH_TOKEN", "")
        self._access: str | None = None
        self._exp = 0.0
        if not (self.token_url and self.client_id and self.refresh_token):
            raise RuntimeError("Missing FPL_TOKEN_URL / FPL_CLIENT_ID / FPL_REFRESH_TOKEN in .env")

    def invalidate(self):
        self._access, self._exp = None, 0.0

    def access_token(self) -> str:
        if self._access and time.time() < self._exp - 60:
            return self._access
        r = httpx.post(
            self.token_url,
            data={
                "grant_type": "refresh_token",
                "refresh_token": self.refresh_token,
                "client_id": self.client_id,
            },
            timeout=30,
        )
        r.raise_for_status()
        tok = r.json()
        self._access = tok["access_token"]
        self._exp = time.time() + int(tok.get("expires_in", 3600))
        new_rt = tok.get("refresh_token")
        if new_rt and new_rt != self.refresh_token:  # PingOne rotates refresh tokens
            self.refresh_token = new_rt
            if ENV_FILE.exists():
                set_key(str(ENV_FILE), "FPL_REFRESH_TOKEN", new_rt)
        return self._access
