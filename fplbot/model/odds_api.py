"""Optional: upcoming EPL odds from The Odds API -> fixture-level market λ.

Off unless the environment variable ODDS_API_KEY is set. One request (regions=uk,
markets=h2h,totals) costs 2 credits; responses are cached under DATA_DIR/cache for
`cache_hours` (default 12h → at most ~28 credits a week even if the bot runs hourly).
The key is sent as a query parameter, so errors are logged without the URL and the key is never
written to the cache or printed.
"""
from __future__ import annotations

import calendar
import json
import logging
import os
import re
import time
from pathlib import Path

import numpy as np

from fplbot.model.odds import implied_lambdas

log = logging.getLogger(__name__)

URL = "https://api.the-odds-api.com/v4/sports/soccer_epl/odds"

# Odds API name (normalised) -> FPL bootstrap name (lowercased).
ALIASES = {
    "brighton and hove albion": "brighton",
    "manchester city": "man city",
    "manchester united": "man utd",
    "newcastle united": "newcastle",
    "nottingham forest": "nott'm forest",
    "tottenham hotspur": "spurs",
    "west ham united": "west ham",
    "wolverhampton wanderers": "wolves",
    "leeds united": "leeds",
    "leicester city": "leicester",
    "sheffield united": "sheffield utd",
    "sheffield wednesday": "sheffield wed",
    "west bromwich albion": "west brom",
}


def _norm(name: str) -> str:
    n = re.sub(r"\s+", " ", str(name).lower().replace("&", "and")).strip()
    return " ".join(t for t in n.split(" ") if t not in ("fc", "afc"))


def map_odds_names(bootstrap: dict, names) -> dict[str, int]:
    """Odds API team names -> FPL team id. Unknown names are left out (never guessed)."""
    canon = {}
    for t in bootstrap["teams"]:
        for k in (t["name"], t.get("short_name", "")):
            canon[_norm(k)] = t["id"]
    out = {}
    for n in set(names):
        k = _norm(n)
        hit = canon.get(ALIASES.get(k, k))
        if hit is None:  # "Coventry City" vs FPL "Coventry" (or the reverse)
            for suffix in (" city", " town", " united"):
                hit = hit or canon.get(k + suffix) or (canon.get(k[: -len(suffix)]) if k.endswith(suffix) else None)
        if hit is not None:
            out[n] = hit
    return out


def event_lambdas(event: dict) -> tuple[float, float] | None:
    """One Odds API event -> (λ_home, λ_away), averaging de-vigged probabilities over bookmakers."""
    home, away = event.get("home_team"), event.get("away_team")
    p1x2, pov = [], []
    for bk in event.get("bookmakers", []):
        for m in bk.get("markets", []):
            oc = m.get("outcomes", [])
            if m.get("key") == "h2h":
                price = {o.get("name"): o.get("price") for o in oc}
                trip = [price.get(home), price.get("Draw"), price.get(away)]
                if all(isinstance(x, (int, float)) and x > 1 for x in trip):
                    inv = 1 / np.array(trip, float)
                    p1x2.append(inv / inv.sum())
            elif m.get("key") == "totals":
                price = {o.get("name"): o.get("price") for o in oc if o.get("point") == 2.5}
                pair = [price.get("Over"), price.get("Under")]
                if all(isinstance(x, (int, float)) and x > 1 for x in pair):
                    inv = 1 / np.array(pair, float)
                    pov.append(inv[0] / inv.sum())
    if not p1x2 or not pov:
        return None
    p = np.mean(p1x2, axis=0)
    lh, la = implied_lambdas([p[0]], [p[2]], [float(np.mean(pov))])
    return float(lh[0]), float(la[0])


def fetch_events(cfg: dict | None = None, cache_dir: Path | None = None, now: float | None = None) -> list[dict]:
    """Cached upcoming EPL events, or [] when ODDS_API_KEY is unset or the request fails."""
    cfg = cfg or {}
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        return []
    if cache_dir is None:
        from fplbot.config import DATA_DIR
        cache_dir = DATA_DIR / "cache"
    path = Path(cache_dir) / "odds_api_soccer_epl.json"
    now = time.time() if now is None else now
    cached = None
    if path.exists():
        try:
            cached = json.loads(path.read_text())
        except ValueError:
            cached = None
    if cached and now - cached.get("fetched_at", 0) < float(cfg.get("cache_hours", 12)) * 3600:
        return cached.get("events", [])
    try:
        import httpx
        r = httpx.get(URL, timeout=30, params={"apiKey": key, "regions": cfg.get("regions", "uk"),
                                               "markets": "h2h,totals", "oddsFormat": "decimal"})
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}")
        events = r.json()
        log.info("Odds API: %d events, %s credits remaining", len(events), r.headers.get("x-requests-remaining"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"fetched_at": now, "events": events}))
        return events
    except Exception as e:  # noqa: BLE001 - never surface the URL (it carries the key)
        log.warning("Odds API fetch failed (%s: %s); %s", type(e).__name__, str(e).replace(key, "***"),
                    "using stale cache" if cached else "no market blend")
        return (cached or {}).get("events", [])


def upcoming_market_lambdas(bootstrap: dict, cfg: dict | None = None, cache_dir: Path | None = None,
                            events: list[dict] | None = None, now: float | None = None) -> dict:
    """{(home_id, away_id): (λ_home, λ_away)} for upcoming EPL fixtures with odds. {} without a key."""
    if events is None:
        events = fetch_events(cfg, cache_dir, now)
    if not events:
        return {}
    now = time.time() if now is None else now
    ids = map_odds_names(bootstrap, {e.get(k) for e in events for k in ("home_team", "away_team")})
    out = {}
    for e in events:
        try:
            ts = calendar.timegm(time.strptime(e["commence_time"], "%Y-%m-%dT%H:%M:%SZ"))
        except (KeyError, ValueError):
            continue
        h, a = ids.get(e.get("home_team")), ids.get(e.get("away_team"))
        if ts <= now or h is None or a is None:
            continue
        lam = event_lambdas(e)
        if lam is not None:
            out[(h, a)] = lam
    missing = {e.get("home_team") for e in events} | {e.get("away_team") for e in events}
    missing -= set(ids)
    if missing:
        log.warning("Odds API teams not mapped to FPL (skipped): %s", sorted(map(str, missing)))
    return out
