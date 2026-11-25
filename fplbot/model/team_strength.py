"""Time-decayed Poisson (Dixon-Coles style) team strength model.

FPL's own `strength_attack_*` / `strength_defence_*` are 0 for every team in the live API, so
we fit our own ratings from match results (football-data.co.uk CSVs):

  log λ_home = μ + h + att_H − def_A
  log λ_away = μ     + att_A − def_H

The target per side is a blend of up to three sources,
  y = (w_goals·goals + w_xg·xG + w_odds·market_λ) / (sum of the weights of the sources present),
so a season without xG (2025/26) or a match without odds just renormalises over what exists.
market_λ is the betting market's implied expected goals for that side (see `odds.py`).

We minimise the weighted Poisson quasi-likelihood  Σ w·(λ − y·log λ)  with w = exp(−ξ·days_ago),
plus a ridge penalty  Σ((att − p_att)² + (def − p_def)²) / (2σ²). The prior means p are 0 (league
average) for established clubs and a negative, history-estimated value for clubs promoted this
season (teams in the latest season of the frame that weren't in the season before). Ridge
means the data dominate as matches accrue: the prior's weight on a rating is roughly
1/σ² / (1/σ² + Σ w·λ), i.e. ~60% after 12 matches and ~30% after 38 at σ=0.2.
Ratings are re-centred so the latest season's clubs average 0.

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
    "target_weights": {"goals": 0.3, "xg": 0.35, "odds": 0.35},
    "ridge_sigma": 0.2,     # prior sd of att/def around their prior mean
    "promoted_prior": {"enabled": True, "att": -0.28, "def": -0.21},  # see config.yaml
    "odds_api": {"enabled": True, "blend_weight": 0.5, "cache_hours": 12, "regions": "uk"},
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
# Kept when present (added by normalize_results from football-data odds columns):
# market-implied expected goals per side and the season (start year, e.g. 2026 for 2026/27).
EXTRA_COLUMNS = ["hmkt", "amkt", "season"]


class TeamMappingError(ValueError):
    pass


def _cfg(cfg) -> dict:
    """Accept the full config (dict/Config), or just the `team_model:` block."""
    raw = getattr(cfg, "raw", cfg) or {}
    block = dict((raw.get("team_model", raw) if isinstance(raw, dict) else {}) or {})
    if "goals_weight" in block and "target_weights" not in block:  # legacy two-source config
        g = float(block["goals_weight"])
        block["target_weights"] = {"goals": g, "xg": 1 - g, "odds": 0.0}
    out = {**DEFAULTS, **block}
    out["target_weights"] = {**{"goals": 0.0, "xg": 0.0, "odds": 0.0}, **out["target_weights"]}
    for k in ("promoted_prior", "odds_api"):
        out[k] = {**DEFAULTS[k], **(out.get(k) or {})}
    return out


def season_of(dates) -> pd.Series:
    """Season start year for each date: August onwards belongs to the season starting that year.
    (Not July: 2019/20 finished on 26 July 2020.) Only a fallback; `load_results` labels rows
    from the file's season code."""
    d = pd.to_datetime(pd.Series(dates))
    return (d.dt.year - (d.dt.month < 8)).astype(int)


# ---------------------------------------------------------------- data ----

def parse_results_csv(text: str) -> pd.DataFrame:
    """football-data.co.uk CSV text -> results frame with RESULT_COLUMNS."""
    return normalize_results(pd.read_csv(io.StringIO(text.lstrip("﻿"))))


def normalize_results(df: pd.DataFrame) -> pd.DataFrame:
    """Accept a results frame in RESULT_COLUMNS or football-data.co.uk layout
    (Date, HomeTeam, AwayTeam, FTHG, FTAG, optional HxG/AxG) and return RESULT_COLUMNS."""
    if set(RESULT_COLUMNS) <= set(df.columns):
        out = df[RESULT_COLUMNS + [c for c in EXTRA_COLUMNS if c in df.columns]].copy()
        out["date"] = pd.to_datetime(out["date"])
        for c in ("hmkt", "amkt"):
            if c not in out:
                out[c] = np.nan
        if "season" not in out:
            out["season"] = season_of(out["date"]).to_numpy()
        return out.reset_index(drop=True)
    from fplbot.model.odds import implied_goals
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
    out = pd.concat([out, implied_goals(df)], axis=1)
    out["season"] = season_of(out["date"]).to_numpy()
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
        df = parse_results_csv(path.read_text(encoding="utf-8", errors="replace"))
        df["season"] = 2000 + int(str(season)[:2])
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def map_teams(bootstrap: dict, names, allow_missing: bool = False) -> dict[int, str]:
    """Map every FPL team id to the football-data name used in `names`. Raises if any of the
    bootstrap teams can't be found — never guesses. With `allow_missing`, unmatched teams are
    left out of the result instead (the caller decides what to do with them)."""
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
    if missing and not allow_missing:
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
    promoted: frozenset = frozenset()
    prior_att: float = 0.0  # prior means used for `promoted` (and for unseen teams)
    prior_def: float = 0.0


def detect_promoted(results: pd.DataFrame) -> set[str]:
    """Clubs in the latest season of `results` that weren't in the season before it. Uses only
    the frame passed in; returns an empty set when the previous season isn't in it."""
    df = results if "season" in results else normalize_results(results)
    seasons = sorted(set(df["season"]))
    if len(seasons) < 2 or seasons[-2] != seasons[-1] - 1:
        return set()
    def teams(s):
        d = df[df["season"] == s]
        return set(d["home"]) | set(d["away"])
    return teams(seasons[-1]) - teams(seasons[-2])


def build_target(df: pd.DataFrame, weights: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per-side target: weighted mean of goals, xG and market-implied goals over the sources
    present in each row (weights renormalised per row)."""
    out = []
    for side in ("h", "a"):
        num = np.zeros(len(df))
        den = np.zeros(len(df))
        for key, col in (("goals", f"{side}g"), ("xg", f"{side}xg"), ("odds", f"{side}mkt")):
            wk = float(weights.get(key, 0.0))
            if wk <= 0 or col not in df:
                continue
            v = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
            have = ~np.isnan(v)
            num += np.where(have, wk * np.nan_to_num(v), 0.0)
            den += np.where(have, wk, 0.0)
        goals = df[f"{side}g"].astype(float).to_numpy()
        out.append(np.where(den > 0, num / np.where(den > 0, den, 1.0), goals))
    return out[0], out[1]


def fit_ratings(results: pd.DataFrame, cfg=None, ref_date=None, promoted=None) -> Ratings:
    """Fit ratings. `promoted` overrides detection (an iterable of names; empty to disable)."""
    c = _cfg(cfg)
    df = normalize_results(results)
    teams = sorted(set(df["home"]) | set(df["away"]))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    hi = df["home"].map(idx).to_numpy()
    ai = df["away"].map(idx).to_numpy()

    yh, ya = build_target(df, c["target_weights"])

    ref = pd.Timestamp(ref_date) if ref_date is not None else df["date"].max()
    days = (ref - pd.to_datetime(df["date"])).dt.days.clip(lower=0).to_numpy(float)
    w = np.exp(-float(c["xi"]) * days)
    lam = 1.0 / float(c["ridge_sigma"]) ** 2

    pp = c["promoted_prior"]
    if promoted is None:
        promoted = detect_promoted(df) if pp.get("enabled", True) else set()
    promoted = frozenset(t for t in promoted if t in idx)
    pa_mean, pd_mean = float(pp.get("att", 0.0)), float(pp.get("def", 0.0))
    prior_a, prior_d = np.zeros(n), np.zeros(n)
    for t in promoted:
        prior_a[idx[t]], prior_d[idx[t]] = pa_mean, pd_mean

    def unpack(x):
        return x[0], x[1], x[2:2 + n], x[2 + n:]

    def f(x):
        mu, h, att, dfn = unpack(x)
        eta_h = mu + h + att[hi] - dfn[ai]
        eta_a = mu + att[ai] - dfn[hi]
        lh, la = np.exp(eta_h), np.exp(eta_a)
        ra_, rd_ = att - prior_a, dfn - prior_d
        nll = np.sum(w * (lh - yh * eta_h)) + np.sum(w * (la - ya * eta_a))
        nll += 0.5 * lam * (ra_ @ ra_ + rd_ @ rd_)
        rh, ra = w * (lh - yh), w * (la - ya)  # d/d eta
        grad = np.zeros_like(x)
        grad[0] = rh.sum() + ra.sum()
        grad[1] = rh.sum()
        ga = np.bincount(hi, rh, n) + np.bincount(ai, ra, n) + lam * ra_
        gd = -np.bincount(ai, rh, n) - np.bincount(hi, ra, n) + lam * rd_
        grad[2:2 + n], grad[2 + n:] = ga, gd
        return nll, grad

    mean_y = max(1e-3, (np.sum(w * yh) + np.sum(w * ya)) / (2 * w.sum()))
    x0 = np.zeros(2 + 2 * n)
    x0[0] = math.log(mean_y)
    x0[2:2 + n], x0[2 + n:] = prior_a, prior_d
    res = minimize(f, x0, jac=True, method="L-BFGS-B", options={"maxiter": 2000, "gtol": 1e-9})
    if not res.success:
        log.warning("team model fit did not fully converge: %s", res.message)
    mu, h, att, dfn = unpack(res.x)
    # Re-centre on the latest season's clubs (shifts are absorbed by μ), so 0 = this season's
    # average club and the promoted prior is measured on the same scale as when it was estimated.
    cur = df[df["season"] == df["season"].max()]
    cur_idx = [idx[t] for t in sorted(set(cur["home"]) | set(cur["away"]))]
    ma, md = att[cur_idx].mean(), dfn[cur_idx].mean()
    att, dfn, mu = att - ma, dfn - md, mu + ma - md
    counts = pd.concat([df["home"], df["away"]]).value_counts()
    return Ratings(float(mu), float(h),
                   {t: float(att[i]) for t, i in idx.items()},
                   {t: float(dfn[i]) for t, i in idx.items()},
                   {t: int(counts.get(t, 0)) for t in teams},
                   promoted, pa_mean - ma, pd_mean - md)


class TeamModel:
    """Fitted ratings bound to FPL team ids.

    `unseen` are FPL teams with no results yet (e.g. promoted clubs before their first match);
    they get the promoted prior. `market` maps (home_id, away_id) -> (λ_home, λ_away) from
    upcoming-fixture odds; when set, expected_goals for that fixture is the geometric blend
    λ_model^(1−w) · λ_market^w with w = `market_weight`.
    """

    def __init__(self, ratings: Ratings, id_to_name: dict[int, str], unseen: dict[int, str] | None = None,
                 market: dict[tuple[int, int], tuple[float, float]] | None = None, market_weight: float = 0.0):
        self.ratings = ratings
        self.id_to_name = dict(id_to_name)
        self.unseen = dict(unseen or {})
        self.market = dict(market or {})
        self.market_weight = float(market_weight)

    @classmethod
    def from_results(cls, results: pd.DataFrame, bootstrap: dict, cfg=None, ref_date=None,
                     unseen_as_promoted: bool = False) -> "TeamModel":
        results = normalize_results(results)
        mapping = map_teams(bootstrap, set(results["home"]) | set(results["away"]),
                            allow_missing=unseen_as_promoted)
        unseen = {t["id"]: t["name"] for t in bootstrap["teams"] if t["id"] not in mapping}
        if unseen:
            log.warning("no results for FPL teams %s; using the promoted-club prior", sorted(unseen.values()))
        return cls(fit_ratings(results, cfg, ref_date), mapping, unseen)

    def att(self, team_id: int) -> float:
        if team_id in self.unseen:
            return self.ratings.prior_att
        return self.ratings.att[self.id_to_name[team_id]]

    def dfn(self, team_id: int) -> float:
        if team_id in self.unseen:
            return self.ratings.prior_def
        return self.ratings.dfn[self.id_to_name[team_id]]

    def model_expected_goals(self, team_id: int, opp_id: int, home: bool) -> float:
        r = self.ratings
        return math.exp(r.mu + (r.home if home else 0.0) + self.att(team_id) - self.dfn(opp_id))

    def expected_goals(self, team_id: int, opp_id: int, home: bool) -> float:
        lam = self.model_expected_goals(team_id, opp_id, home)
        mk = self.market.get((team_id, opp_id) if home else (opp_id, team_id))
        if mk is None or self.market_weight <= 0:
            return lam
        m = mk[0] if home else mk[1]
        w = self.market_weight
        return math.exp((1 - w) * math.log(lam) + w * math.log(m))

    def p_clean_sheet(self, team_id: int, opp_id: int, home: bool) -> float:
        return math.exp(-self.expected_goals(opp_id, team_id, not home))

    @property
    def league_avg(self) -> float:
        """Expected goals per team per match for two average teams (home/away averaged)."""
        return 0.5 * (math.exp(self.ratings.mu + self.ratings.home) + math.exp(self.ratings.mu))

    def table(self) -> pd.DataFrame:
        r = self.ratings
        rows = [{"id": i, "team": n, "att": r.att[n], "def": r.dfn[n], "n": r.n_matches[n],
                 "promoted": n in r.promoted} for i, n in self.id_to_name.items()]
        rows += [{"id": i, "team": n, "att": r.prior_att, "def": r.prior_def, "n": 0, "promoted": True}
                 for i, n in self.unseen.items()]
        return pd.DataFrame(rows).set_index("id")


def fit_team_model(bootstrap: dict, cfg=None, results: pd.DataFrame | None = None,
                   offline: bool = False, cache_dir: Path | None = None,
                   unseen_as_promoted: bool = False, market: bool | None = None) -> TeamModel:
    """Fit the team model. Pass `results` (RESULT_COLUMNS or football-data frame) to skip
    downloading; the fit then uses only those rows.

    `unseen_as_promoted`: FPL teams with no rows in `results` get the promoted prior instead of
    raising (useful for backtests at GW1, before promoted clubs have played).
    `market`: blend upcoming-fixture odds from The Odds API (needs ODDS_API_KEY). Default: only
    when we loaded live results ourselves (results None, not offline) and `odds_api.enabled`.
    It is a no-op without the key.
    """
    live = results is None and not offline
    if results is None:
        results = load_results(cfg, offline=offline, cache_dir=cache_dir)
    tm = TeamModel.from_results(results, bootstrap, cfg, unseen_as_promoted=unseen_as_promoted)
    c = _cfg(cfg)
    oc = c.get("odds_api") or {}
    if market is None:
        market = live and bool(oc.get("enabled", True))
    if market:
        from fplbot.model.odds_api import upcoming_market_lambdas
        tm.market = upcoming_market_lambdas(bootstrap, oc, cache_dir=cache_dir)
        tm.market_weight = float(oc.get("blend_weight", 0.5)) if tm.market else 0.0
    return tm
