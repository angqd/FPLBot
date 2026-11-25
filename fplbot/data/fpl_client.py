"""Thin client for the (unofficial, undocumented) FPL JSON API.

Public endpoints need no auth. Everything the model needs is here:
  bootstrap-static/          players (price, form, xG/xA per 90, DefCon/90, status, news, ep_next),
                             teams (home/away attack/defence strength), gameweeks (deadlines)
  fixtures/                  every fixture with event (GW), home/away, FDR
  element-summary/{id}/      per-player GW-by-GW history + upcoming fixtures
  event/{gw}/live/           live/final points per player for a GW
Authenticated (see auth.py):
  my-team/{entry}/           current picks with *selling prices*, bank, free transfers
  transfers/                 POST to make transfers
"""
from __future__ import annotations

import time
from typing import Any

import httpx

BASE = "https://fantasy.premierleague.com/api/"
UA = {"User-Agent": "fplbot/0.1 (+personal project)"}


class FPLClient:
    def __init__(self, auth=None, timeout: float = 30.0, min_interval: float = 0.5):
        self._http = httpx.Client(base_url=BASE, headers=UA, timeout=timeout)
        self._auth = auth
        self._min_interval = min_interval
        self._last = 0.0

    # -- plumbing -------------------------------------------------------------
    def _req(self, method: str, path: str, *, authed: bool = False, **kw) -> Any:
        wait = self._min_interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)  # be polite; FPL rate-limits aggressive clients
        headers = kw.pop("headers", {})
        if authed:
            if not self._auth:
                raise RuntimeError("This endpoint needs auth; configure FPL_REFRESH_TOKEN")
            headers["X-API-Authorization"] = f"Bearer {self._auth.access_token()}"
        for attempt in range(4):
            r = self._http.request(method, path, headers=headers, **kw)
            self._last = time.monotonic()
            if r.status_code in (401, 403) and authed and attempt == 0:
                self._auth.invalidate()
                headers["X-API-Authorization"] = f"Bearer {self._auth.access_token()}"
                continue
            if r.status_code in (429, 502, 503):
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            return r.json() if r.content else None
        r.raise_for_status()

    # -- public ---------------------------------------------------------------
    def bootstrap(self) -> dict:
        return self._req("GET", "bootstrap-static/")

    def fixtures(self) -> list[dict]:
        return self._req("GET", "fixtures/")

    def element_summary(self, player_id: int) -> dict:
        return self._req("GET", f"element-summary/{player_id}/")

    def live(self, gw: int) -> dict:
        return self._req("GET", f"event/{gw}/live/")

    def entry_picks(self, entry: int, gw: int) -> dict:
        """Public view of a team's picks for a past/current GW (no selling prices)."""
        return self._req("GET", f"entry/{entry}/event/{gw}/picks/")

    # -- authenticated --------------------------------------------------------
    def my_team(self, entry: int) -> dict:
        """picks[] with selling_price/purchase_price, transfers{bank, limit, made, ...}, chips."""
        return self._req("GET", f"my-team/{entry}/", authed=True)

    def post_transfers(self, payload: dict) -> Any:
        return self._req("POST", "transfers/", authed=True, json=payload)

    def post_lineup(self, entry: int, payload: dict) -> Any:
        return self._req("POST", f"my-team/{entry}/", authed=True, json=payload)
