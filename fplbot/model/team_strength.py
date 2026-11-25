"""Time-decayed Poisson (Dixon-Coles style) team strength model.

FPL's own `strength_attack_*` / `strength_defence_*` are 0 for every team in the live API, so
we fit our own ratings from match results (football-data.co.uk CSVs):

  log λ_home = μ + h + att_H − def_A
  log λ_away = μ     + att_A − def_H

The target per side is y = w_g·goals + (1−w_g)·xG (goals only when xG is missing). We minimise
the weighted Poisson quasi-likelihood  Σ w·(λ − y·log λ)  with w = exp(−ξ·days_ago), plus a
ridge penalty  Σ(att² + def²) / (2σ²)  that shrinks teams with few matches (promoted clubs)
toward league average. μ and h are unpenalised, so at the optimum Σatt = Σdef = 0; we
re-centre explicitly afterwards anyway.

No Dixon-Coles ρ term: its low-score correction leaves the marginal P(team scores 0)
unchanged, so it doesn't affect P(clean sheet), which is all the xPts model needs from it.
"""
from __future__ import annotations

import io
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

log = logging.getLogger(__name__)

DEFAULTS = {
    "league": "E0",
    "seasons": ["2526", "2627"],
    "base_url": "https://www.football-data.co.uk/mmz4281",
    "cache_max_age_hours": 12,
    "xi": 0.0019,           # time decay per day (half-life ≈ 1 year)
    "goals_weight": 0.5,    # y = goals_weight·goals + (1−goals_weight)·xG
    "ridge_sigma": 0.2,     # prior sd of att/def around 0 (league average)
}

# football-data.co.uk name -> FPL bootstrap `teams[].name`. Names that already match
# (Arsenal, Man City, Nott'm Forest, Wolves, Leeds, Sunderland, ...) need no entry.
ALIASES = {
    "Man United": "Man Utd",
    "Tottenham": "Spurs",
    "Coventry": "Coventry City",
    "Hull": "Hull City",
    "Ipswich": "Ipswich Town",
    "Sheffield United": "Sheffield Utd",
    "Sheffield Weds": "Sheffield Wed",
}

RESULT_COLUMNS = ["date", "home", "away", "hg", "ag", "hxg", "axg"]


class TeamMappingError(ValueError):
    pass


def _cfg(cfg) -> dict:
    """Accept the full config (dict/Config), or just the `team_model:` block."""
    raw = getattr(cfg, "raw", cfg) or {}
    block = raw.get("team_model", raw) if isinstance(raw, dict) else {}
    return {**DEFAULTS, **(block or {})}


# ---------------------------------------------------------------- data ----

def parse_results_csv(text: str) -> pd.DataFrame:
    """football-data.co.uk CSV text -> results frame with RESULT_COLUMNS."""
    return normalize_results(pd.read_csv(io.StringIO(text.lstrip("﻿"))))


def normalize_results(df: pd.DataFrame) -> pd.DataFrame:
    """Accept a results frame in RESULT_COLUMNS or football-data.co.uk layout
    (Date, HomeTeam, AwayTeam, FTHG, FTAG, optional HxG/AxG) and return RESULT_COLUMNS."""
    if set(RESULT_COLUMNS) <= set(df.columns):
        out = df[RESULT_COLUMNS].copy()
        out["date"] = pd.to_datetime(out["date"])
        return out.reset_index(drop=True)
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    raw = df["Date"].astype(str).str.strip()
    long = pd.to_datetime(raw.where(raw.str.len() == 10), format="%d/%m/%Y", errors="coerce")
    short = pd.to_datetime(raw.where(raw.str.len() == 8), format="%d/%m/%y", errors="coerce")
    out = pd.DataFrame({
        "date": long.fillna(short),
        "home": df["HomeTeam"].astype(str).str.strip(),
        "away": df["AwayTeam"].astype(str).str.strip(),
        "hg": df["FTHG"].astype(float),
        "ag": df["FTAG"].astype(float),
        "hxg": pd.to_numeric(df["HxG"], errors="coerce") if "HxG" in df else np.nan,
        "axg": pd.to_numeric(df["AxG"], errors="coerce") if "AxG" in df else np.nan,
    })
    if out["date"].isna().any():
        raise ValueError(f"unparseable dates in results CSV: {df['Date'][out['date'].isna()].tolist()[:5]}")
    return out.reset_index(drop=True)


def load_results(cfg=None, offline: bool = False, cache_dir: Path | None = None) -> pd.DataFrame:
    """Download (or read cached) football-data CSVs for the configured seasons.

    A cached file younger than `cache_max_age_hours` is used as-is. If a download fails, a stale
    cache is used with a warning. `offline=True` never touches the network.
    """
    c = _cfg(cfg)
    if cache_dir is None:
        from fplbot.config import DATA_DIR
        cache_dir = DATA_DIR / "cache"
    cache_dir = Path(cache_dir)
    frames = []
    for season in c["seasons"]:
        path = cache_dir / f"football-data_{c['league']}_{season}.csv"
        fresh = path.exists() and (time.time() - path.stat().st_mtime) < c["cache_max_age_hours"] * 3600
        if not fresh and not offline:
            url = f"{c['base_url']}/{season}/{c['league']}.csv"
            try:
                import httpx
                r = httpx.get(url, timeout=30, follow_redirects=True)
                r.raise_for_status()
                parse_results_csv(r.text)  # validate before caching
                cache_dir.mkdir(parents=True, exist_ok=True)
                path.write_text(r.text)
            except Exception as e:  # noqa: BLE001 - fall back to cache below
                if not path.exists():
                    raise RuntimeError(f"can't download {url} and no cache at {path}: {e}") from e
                log.warning("download of %s failed (%s); using stale cache %s", url, e, path)
        if not path.exists():
            raise FileNotFoundError(f"offline and no cached results at {path}")
        frames.append(parse_results_csv(path.read_text()))
    return pd.concat(frames, ignore_index=True)


def map_teams(bootstrap: dict, names) -> dict[int, str]:
    """Map every FPL team id to the football-data name used in `names`. Raises if any of the
    bootstrap teams can't be found — never guesses."""
    canon = {}  # canonical (FPL-style, lowercased) -> football-data name
    for n in set(names):
        canon[ALIASES.get(n, n).lower()] = n
    out, missing = {}, []
    for t in bootstrap["teams"]:
        hit = canon.get(str(t["name"]).lower()) or canon.get(str(t.get("short_name", "")).lower())
        if hit is None:
            missing.append(t["name"])
        else:
            out[t["id"]] = hit
    if missing:
        unused = sorted(set(names) - set(out.values()))
        raise TeamMappingError(
            f"can't map FPL teams {missing} to results team names; unmatched result names: {unused}. "
            "Add an entry to fplbot.model.team_strength.ALIASES.")
    if len(set(out.values())) != len(out):
        raise TeamMappingError(f"two FPL teams mapped to the same results name: {out}")
    return out


# --------------------------------------------------------------- model ----

@dataclass
class Ratings:
    mu: float
    home: float
    att: dict[str, float]
    dfn: dict[str, float]
    n_matches: dict[str, int]


def fit_ratings(results: pd.DataFrame, cfg=None, ref_date=None) -> Ratings:
    c = _cfg(cfg)
    df = normalize_results(results)
    teams = sorted(set(df["home"]) | set(df["away"]))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    hi = df["home"].map(idx).to_numpy()
    ai = df["away"].map(idx).to_numpy()

    g = float(c["goals_weight"])
    hxg = df["hxg"].astype(float).to_numpy()
    axg = df["axg"].astype(float).to_numpy()
    hg, ag = df["hg"].astype(float).to_numpy(), df["ag"].astype(float).to_numpy()
    yh = np.where(np.isnan(hxg), hg, g * hg + (1 - g) * np.nan_to_num(hxg))
    ya = np.where(np.isnan(axg), ag, g * ag + (1 - g) * np.nan_to_num(axg))

    ref = pd.Timestamp(ref_date) if ref_date is not None else df["date"].max()
    days = (ref - pd.to_datetime(df["date"])).dt.days.clip(lower=0).to_numpy(float)
    w = np.exp(-float(c["xi"]) * days)
    lam = 1.0 / float(c["ridge_sigma"]) ** 2

    def unpack(x):
        return x[0], x[1], x[2:2 + n], x[2 + n:]

    def f(x):
        mu, h, att, dfn = unpack(x)
        eta_h = mu + h + att[hi] - dfn[ai]
        eta_a = mu + att[ai] - dfn[hi]
        lh, la = np.exp(eta_h), np.exp(eta_a)
        nll = np.sum(w * (lh - yh * eta_h)) + np.sum(w * (la - ya * eta_a))
        nll += 0.5 * lam * (att @ att + dfn @ dfn)
        rh, ra = w * (lh - yh), w * (la - ya)  # d/d eta
        grad = np.zeros_like(x)
        grad[0] = rh.sum() + ra.sum()
        grad[1] = rh.sum()
        ga = np.bincount(hi, rh, n) + np.bincount(ai, ra, n) + lam * att
        gd = -np.bincount(ai, rh, n) - np.bincount(hi, ra, n) + lam * dfn
        grad[2:2 + n], grad[2 + n:] = ga, gd
        return nll, grad

    mean_y = max(1e-3, (np.sum(w * yh) + np.sum(w * ya)) / (2 * w.sum()))
    x0 = np.zeros(2 + 2 * n)
    x0[0] = math.log(mean_y)
    res = minimize(f, x0, jac=True, method="L-BFGS-B", options={"maxiter": 2000, "gtol": 1e-9})
    if not res.success:
        log.warning("team model fit did not fully converge: %s", res.message)
    mu, h, att, dfn = unpack(res.x)
    # explicit sum-to-zero (shifts are absorbed by μ; ridge already makes them ~0)
    ma, md = att.mean(), dfn.mean()
    att, dfn, mu = att - ma, dfn - md, mu + ma - md
    counts = pd.concat([df["home"], df["away"]]).value_counts()
    return Ratings(float(mu), float(h),
                   {t: float(att[i]) for t, i in idx.items()},
                   {t: float(dfn[i]) for t, i in idx.items()},
                   {t: int(counts.get(t, 0)) for t in teams})


class TeamModel:
    """Fitted ratings bound to FPL team ids."""

    def __init__(self, ratings: Ratings, id_to_name: dict[int, str]):
        self.ratings = ratings
        self.id_to_name = dict(id_to_name)

    @classmethod
    def from_results(cls, results: pd.DataFrame, bootstrap: dict, cfg=None, ref_date=None) -> "TeamModel":
        results = normalize_results(results)
        mapping = map_teams(bootstrap, set(results["home"]) | set(results["away"]))
        return cls(fit_ratings(results, cfg, ref_date), mapping)

    def att(self, team_id: int) -> float:
        return self.ratings.att[self.id_to_name[team_id]]

    def dfn(self, team_id: int) -> float:
        return self.ratings.dfn[self.id_to_name[team_id]]

    def expected_goals(self, team_id: int, opp_id: int, home: bool) -> float:
        r = self.ratings
        return math.exp(r.mu + (r.home if home else 0.0) + self.att(team_id) - self.dfn(opp_id))

    def p_clean_sheet(self, team_id: int, opp_id: int, home: bool) -> float:
        return math.exp(-self.expected_goals(opp_id, team_id, not home))

    @property
    def league_avg(self) -> float:
        """Expected goals per team per match for two average teams (home/away averaged)."""
        return 0.5 * (math.exp(self.ratings.mu + self.ratings.home) + math.exp(self.ratings.mu))

    def table(self) -> pd.DataFrame:
        rows = [{"id": i, "team": n, "att": self.ratings.att[n], "def": self.ratings.dfn[n],
                 "n": self.ratings.n_matches[n]} for i, n in self.id_to_name.items()]
        return pd.DataFrame(rows).set_index("id")


def fit_team_model(bootstrap: dict, cfg=None, results: pd.DataFrame | None = None,
                   offline: bool = False, cache_dir: Path | None = None) -> TeamModel:
    """Fit the team model. Pass `results` (RESULT_COLUMNS frame) to skip downloading."""
    if results is None:
        results = load_results(cfg, offline=offline, cache_dir=cache_dir)
    return TeamModel.from_results(results, bootstrap, cfg)
