"""Deterministic expected-points (xPts) model.

For each player and each upcoming GW, per fixture (doubles sum, blanks give 0):

  Minutes model (`minutes_profile` + `availability`), from bootstrap-static only:
    avail      P(fit and in the squad) from status / chance_of_playing / news return dates
    P(start)   = avail * shrunk start rate (starts / team games since joining the club)
    E[m|start] = shrunk minutes per start;   P(sub) = avail * P(sub | not started)
    P(60+)     = P(start) * P(60+ | start),  P(60+ | start) a linear map of E[m|start]
    xMins      = P(start) * E[m|start] + P(sub) * sub_mins

  Per-90 rates are shrunk toward a position x price-band prior (empirical Bayes):
    post = (mins * rate + k * prior) / (mins + k),  k per stat in config (`shrinkage.k`).

  xPts = appearance + attack + clean_sheet + defcon + saves + bonus - conceded
    appearance  = 2 * P(60+) + 1 * (P(appear) - P(60+))
    attack      = (xG90 * goal_pts + xA90 * 3) * xMins/90 * atk_mult
                  atk_mult = team_xg(this fixture) / team_xg(vs average opponent), so team
                  quality already in the player's xG isn't counted twice.
    clean_sheet = P(60+) * P(CS)^(m/90) * cs_pts      (Poisson: no goal while on the pitch)
    conceded    = -P(start) * E[floor(G/2)],  G ~ Poisson(opp_xg * m/90)      GK/DEF
    saves       = P(start) * E[floor(S/3)],   S ~ Poisson(saves90 * m/90 * opp_mult)   GK
    defcon      = 2 * [P(start) * P(N_start >= thr) + P(sub) * P(N_sub >= thr)],
                  N ~ Poisson (or neg. binomial if `defcon.dispersion` > 0) with
                  mean = shrunk DC90 * minutes/90.  thr: DEF 10 (CBIT), MID/FWD 12 (CBIRT).
    bonus       = w * shrunk bonus90 * xMins/90 + (1-w) * event_bonus,  w = `bonus.blend`
                  event_bonus = linear in E[goals], E[assists], P(CS & 60+), P(DefCon), E[saves],
                  P(60+); coefficients fitted on 2026/27 per-match data (post BPS changes).

GW+1 is blended with FPL's own `ep_next` (weight `blend_fpl_ep`; 0 = pure model).

Every term is inspectable; a learned model can replace this behind the same `project()`.
"""
from __future__ import annotations

import bisect
import copy
import math
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import pandas as pd

DEFAULT_SCORING = {
    "goal": {1: 10, 2: 6, 3: 5, 4: 4},
    "assist": 3,
    "cs": {1: 4, 2: 4, 3: 1, 4: 0},
    "defcon": {1: 0, 2: 2, 3: 2, 4: 2},
}
POS_NAME = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
NAME_POS = {v: k for k, v in POS_NAME.items()}

# Fallbacks for every model knob; config.yaml's `model:` block overrides (deep-merged).
DEFAULT_MODEL_CFG = {
    "blend_fpl_ep": 0.35,
    "minutes": {
        "start_prior": 0.40,        # P(start) prior for a player with no eligible team games
        "start_prior_games": 1.0,   # k (games) for the start-rate shrinkage
        "start_mins_prior": 75.0,   # E[minutes | start] prior
        "start_mins_k": 2.0,        # k (starts) for E[minutes | start]
        "sub_mins": 22.0,           # average minutes of a sub appearance
        "sub_prior": 0.35,          # P(sub appearance | fit, not started) prior
        "sub_prior_games": 3.0,
        "p60_intercept": 0.024,      # P(60+ | start) = clip(a + b * E[m|start])
        "p60_slope": 0.0111,
        "doubtful_default": 0.5,    # 'd' status with no chance % given
        "recovery_per_gw": 0.15,    # unknown-return injury: availability gained per GW
        "suspension_recovery_per_gw": 0.5,
        "return_fitness": 0.8,      # availability from a dated return ("Expected back 20 Dec")
    },
    "shrinkage": {
        "price_bands": [5.0, 6.0, 7.5, 9.5],   # now_cost (£m) band edges for the priors
        "prior_min_minutes": 1800,  # band needs this many minutes, else position prior
        "k": {"xg": 600, "xa": 600, "dc": 270, "saves": 450, "bonus": 900},
    },
    "defcon": {
        "threshold": {"DEF": 10, "MID": 12, "FWD": 12},
        "dispersion": 0.0,          # NB size r (var = mu + mu^2/r); 0 = Poisson (fit best on 2026/27)
    },
    "bonus": {
        "blend": 0.5,               # weight on season bonus/90 (a); 1 - blend on event model (b)
        "event": {                  # E[bonus] per unit of each event, by position
            # Non-negative least squares of per-match bonus on actual events, 2026/27 GW1-11
            # (3083 player-matches, so it already reflects the 2026/27 BPS changes). GK goal: no
            # data, DEF value used.
            "GKP": {"goal": 1.13, "assist": 1.07, "cs": 0.85, "saves": 0.038, "defcon": 0.0, "p60": 0.0},
            "DEF": {"goal": 1.13, "assist": 0.45, "cs": 0.48, "saves": 0.0, "defcon": 0.05, "p60": 0.0},
            "MID": {"goal": 1.38, "assist": 0.51, "cs": 0.01, "saves": 0.0, "defcon": 0.05, "p60": 0.023},
            "FWD": {"goal": 1.60, "assist": 0.14, "cs": 0.05, "saves": 0.0, "defcon": 0.0, "p60": 0.011},
        },
    },
}


def _f(x, default=0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def model_config(cfg: dict | None) -> dict:
    """The `model:` block with every missing key filled from DEFAULT_MODEL_CFG."""
    return _merge(DEFAULT_MODEL_CFG, cfg or {})


def scoring_from_bootstrap(bootstrap: dict) -> dict:
    """Points per event from `game_config.scoring` when present, else DEFAULT_SCORING."""
    s = (bootstrap.get("game_config") or {}).get("scoring") or {}
    out = copy.deepcopy(DEFAULT_SCORING)
    for key, src in (("goal", "goals_scored"), ("cs", "clean_sheets"), ("defcon", "defensive_contribution")):
        if isinstance(s.get(src), dict):
            out[key] = {NAME_POS[n]: int(v) for n, v in s[src].items() if n in NAME_POS}
    if "assists" in s:
        out["assist"] = int(s["assists"])
    return out


# ---------------------------------------------------------------- distributions

def _pois_pmf(k: int, lam: float) -> float:
    return math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1)) if lam > 0 else float(k == 0)


def count_tail(mean: float, thr: int, dispersion: float = 0.0) -> float:
    """P(N >= thr) for N ~ Poisson(mean), or NegBin(mean, size=dispersion) if dispersion > 0."""
    if mean <= 0:
        return 0.0 if thr > 0 else 1.0
    if dispersion and dispersion > 0:
        r = dispersion
        p = r / (r + mean)  # NB(r, p): P(k) = C(k+r-1, k) p^r (1-p)^k, mean r(1-p)/p
        cdf = sum(math.exp(math.lgamma(k + r) - math.lgamma(r) - math.lgamma(k + 1)
                           + r * math.log(p) + k * math.log1p(-p)) for k in range(thr))
    else:
        cdf = sum(_pois_pmf(k, mean) for k in range(thr))
    return max(0.0, 1.0 - cdf)


def expected_floor_div(lam: float, n: int) -> float:
    """E[floor(X / n)] for X ~ Poisson(lam): FPL's '1 pt per 3 saves', '-1 per 2 conceded'."""
    if lam <= 0:
        return 0.0
    # floor(X/n) = sum_{j>=1} 1[X >= j*n]
    return sum(count_tail(lam, j * n) for j in range(1, int(lam / n + 8 * math.sqrt(lam / n + 1)) + 2))


# ---------------------------------------------------------------- minutes model

@dataclass
class MinutesProfile:
    """Per-match minutes for a fit, selected player (availability applied separately)."""
    start_rate: float   # P(start | available)
    e_start: float      # E[minutes | start]
    sub_rate: float     # P(sub appearance | available, not started)
    p60_start: float    # P(60+ minutes | start)
    eligible_games: int


def team_games(fixtures: list[dict], start_gw: int) -> dict[int, list[str | None]]:
    """Kickoff times of each team's played league games before start_gw."""
    out: dict[int, list] = {}
    for fx in fixtures:
        ev = fx.get("event")
        if ev is None or ev >= start_gw or fx.get("finished") is False:
            continue
        for t in (fx["team_h"], fx["team_a"]):
            out.setdefault(t, []).append(fx.get("kickoff_time"))
    return out


def _eligible_games(p: dict, kickoffs: list, fallback: int) -> int:
    """Team games since the player joined (new signings shouldn't be judged on games they missed)."""
    if not kickoffs:
        return fallback
    joined = (p.get("team_join_date") or "")[:10]
    if not joined:
        return len(kickoffs)
    return sum(1 for k in kickoffs if not k or k[:10] >= joined)


def minutes_profile(p: dict, games: int, mc: dict) -> MinutesProfile:
    mins, starts = _f(p.get("minutes")), _f(p.get("starts"))
    games = max(games, int(starts))
    sub_m = mc["sub_mins"]
    bench_games = max(0.0, games - starts)
    # Bootstrap only has season totals, so sub minutes can't be separated exactly: assume the
    # prior sub rate in non-start games, attribute the rest to starts (clipped to 30-90), and
    # whatever minutes starts can't absorb (raw capped at 90) are extra sub appearances.
    if starts:
        raw = (mins - sub_m * mc["sub_prior"] * bench_games) / starts
        raw = min(90.0, max(30.0, raw))
    else:
        raw = mc["start_mins_prior"]
    e_start = (starts * raw + mc["start_mins_k"] * mc["start_mins_prior"]) / (starts + mc["start_mins_k"])
    # Start rate shrunk toward a prior with k pseudo-games: 11/11 -> ~0.95, which matches the
    # holdout start rate of 'nailed' players (0.94, 2026/27 per-match data).
    k = mc["start_prior_games"]
    start_rate = (starts + k * mc["start_prior"]) / (games + k)
    sub_apps = max(0.0, mins - starts * raw) / sub_m
    ks = mc["sub_prior_games"]
    sub_rate = min(1.0, (sub_apps + ks * mc["sub_prior"]) / (bench_games + ks))
    p60 = min(1.0, max(0.0, mc["p60_intercept"] + mc["p60_slope"] * e_start))
    return MinutesProfile(min(1.0, start_rate), e_start, sub_rate, p60, games)


_RETURN_RE = re.compile(r"(?:expected back|suspended until)\s+(\d{1,2})\s+([A-Za-z]{3})", re.I)
_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


_WEEKS_RE = re.compile(r"back in (\d{1,2}) weeks?", re.I)


def return_date(news: str, ref: date) -> date | None:
    """'Expected back 20 Dec' / 'Suspended until 5 Dec' -> that date (year inferred from ref);
    'expected back in 3 weeks' -> ref + 3 weeks. None when no date is given."""
    w = _WEEKS_RE.search(news or "")
    if w:
        return ref + timedelta(weeks=int(w.group(1)))
    m = _RETURN_RE.search(news or "")
    if not m or m.group(2).lower() not in _MONTHS:
        return None
    day, mon = int(m.group(1)), _MONTHS[m.group(2).lower()]
    for year in (ref.year - 1, ref.year, ref.year + 1):
        try:
            d = date(year, mon, day)
        except ValueError:
            continue
        if (d - ref).days > -60:   # news dates are never long past
            return d
    return None


def availability(p: dict, k: int, deadline: date | None, ref: date | None, mc: dict) -> float:
    """P(fit and available) for the k-th GW of the horizon (k=0 is start_gw)."""
    status = p.get("status") or "a"
    if status in ("u", "n"):         # left the club / not eligible: assume out for the horizon
        return 0.0
    chance = p.get("chance_of_playing_next_round")
    if chance is None and status != "a":
        chance = p.get("chance_of_playing_this_round")
    if status == "a" and chance is None:
        return 1.0
    if chance is None:
        c0 = mc["doubtful_default"] if status == "d" else 0.0
    else:
        c0 = _f(chance) / 100
    if k == 0 or c0 >= 1:
        return c0
    if deadline and ref:
        back = return_date(p.get("news", ""), ref)
        if back is not None:
            return max(c0, mc["return_fitness"]) if deadline >= back else c0
    step = mc["suspension_recovery_per_gw"] if status == "s" else mc["recovery_per_gw"]
    return min(1.0, c0 + step * k)


# ---------------------------------------------------------------- shrinkage

_RATE_FIELDS = {  # stat -> (season total field, per-90 field)
    "xg": ("expected_goals", "expected_goals_per_90"),
    "xa": ("expected_assists", "expected_assists_per_90"),
    "dc": ("defensive_contribution", "defensive_contribution_per_90"),
    "saves": ("saves", "saves_per_90"),
    "bonus": ("bonus", None),
}


def _season_total(p: dict, stat: str) -> float:
    total_f, per90_f = _RATE_FIELDS[stat]
    if p.get(total_f) is not None:
        return _f(p.get(total_f))
    return _f(p.get(per90_f)) * _f(p.get("minutes")) / 90 if per90_f else 0.0


def rate_priors(elements: list[dict], sc: dict) -> dict:
    """Minutes-weighted per-90 rate by (position, price band), falling back to position."""
    bands = sc["price_bands"]
    acc: dict = {}
    for p in elements:
        m = _f(p.get("minutes"))
        if m <= 0:
            continue
        b = bisect.bisect_right(bands, p["now_cost"] / 10)
        for key in ((p["element_type"], b), (p["element_type"], None)):
            a = acc.setdefault(key, {"m": 0.0, **{s: 0.0 for s in _RATE_FIELDS}})
            a["m"] += m
            for s in _RATE_FIELDS:
                a[s] += _season_total(p, s)
    pri = {}
    for pos in (1, 2, 3, 4):
        base = acc.get((pos, None))
        pos_rate = {s: (90 * base[s] / base["m"] if base else 0.0) for s in _RATE_FIELDS}
        for b in range(len(bands) + 1):
            a = acc.get((pos, b))
            if a and a["m"] >= sc["prior_min_minutes"]:
                pri[(pos, b)] = {s: 90 * a[s] / a["m"] for s in _RATE_FIELDS}
            else:
                pri[(pos, b)] = dict(pos_rate)
    return pri


def shrunk_rates(p: dict, priors: dict, sc: dict) -> dict:
    """Empirical-Bayes per-90 rates: (mins*rate + k*prior) / (mins + k); 0 mins -> prior."""
    m = _f(p.get("minutes"))
    prior = priors[(p["element_type"], bisect.bisect_right(sc["price_bands"], p["now_cost"] / 10))]
    out = {}
    for s in _RATE_FIELDS:
        k = sc["k"][s]
        out[s] = (90 * _season_total(p, s) + k * prior[s]) / (m + k) if m + k > 0 else prior[s]
    return out


# ---------------------------------------------------------------- team model fallback

def _team_strengths(teams: list[dict]) -> dict[int, dict]:
    out = {}
    for t in teams:
        out[t["id"]] = {
            "att_h": t.get("strength_attack_home", 1100), "att_a": t.get("strength_attack_away", 1100),
            "def_h": t.get("strength_defence_home", 1100), "def_a": t.get("strength_defence_away", 1100),
        }
    att = [v["att_h"] + v["att_a"] for v in out.values()]
    dfn = [v["def_h"] + v["def_a"] for v in out.values()]
    mean_att, mean_def = sum(att) / len(att) / 2, sum(dfn) / len(dfn) / 2
    for v in out.values():
        v["mean_att"], v["mean_def"] = mean_att, mean_def
    return out


def _expected_goals(team: dict, opp: dict, home: bool, cfg: dict) -> float:
    att = team["att_h"] if home else team["att_a"]
    dfn = opp["def_a"] if home else opp["def_h"]
    ha = cfg["home_advantage"] if home else 1 / cfg["home_advantage"]
    ratio = (att / team["mean_att"]) / (dfn / team["mean_def"])
    return cfg["league_avg_goals"] * ratio ** 2.0 * ha  # ^2 widens spread to realistic range


_STRENGTH_KEYS = ("strength_attack_home", "strength_attack_away", "strength_defence_home", "strength_defence_away")


class StrengthTeamModel:
    """Legacy adapter: FPL's strength_* ratings behind the TeamModel interface."""

    def __init__(self, teams: list[dict], cfg: dict):
        self.strengths, self.cfg = _team_strengths(teams), cfg
        self.league_avg = cfg["league_avg_goals"]

    def expected_goals(self, team_id: int, opp_id: int, home: bool) -> float:
        return _expected_goals(self.strengths[team_id], self.strengths[opp_id], home, self.cfg)

    def p_clean_sheet(self, team_id: int, opp_id: int, home: bool) -> float:
        return math.exp(-self.expected_goals(opp_id, team_id, not home))


def _default_team_model(bootstrap: dict, cfg: dict):
    teams = bootstrap["teams"]
    if all(_f(t.get(k)) > 0 for t in teams for k in _STRENGTH_KEYS):
        return StrengthTeamModel(teams, cfg)
    raise ValueError(
        "bootstrap team strength_attack_*/strength_defence_* fields are 0 or missing (as in the live "
        "FPL API), so they can't be used. Pass team_model=fit_team_model(bootstrap, cfg) "
        "(fplbot.model.team_strength) to project().")


# ---------------------------------------------------------------- projection

def _deadlines(bootstrap: dict) -> dict[int, date]:
    out = {}
    for ev in bootstrap.get("events", []):
        try:
            out[ev["id"]] = datetime.fromisoformat(ev["deadline_time"].replace("Z", "+00:00")).date()
        except (KeyError, TypeError, ValueError, AttributeError):
            pass
    return out


def fixture_points(pos: int, mp: MinutesProfile, avail: float, r: dict, team_xg: float, opp_xg: float,
                   p_cs: float, atk_mult: float, opp_mult: float, scoring: dict, mc: dict) -> dict:
    """Expected points (by component) for one fixture. Returns a dict of terms + xmins etc."""
    sub_m = mc["sub_mins"]
    p_start = avail * mp.start_rate
    p_sub = avail * (1 - mp.start_rate) * mp.sub_rate
    p60 = p_start * mp.p60_start
    xmins = p_start * mp.e_start + p_sub * sub_m
    ms = mp.e_start / 90

    goals = r["xg"] * xmins / 90 * atk_mult
    assists = r["xa"] * xmins / 90 * atk_mult
    t = {"appearance": 2 * p60 + 1 * (p_start + p_sub - p60),
         "attack": goals * scoring["goal"][pos] + assists * scoring["assist"]}
    # CS needs 60+ mins and no goal while on the pitch (~ the 60-90 minutes he plays).
    p_cs_on = p_cs ** (min(90.0, max(60.0, mp.e_start)) / 90)
    cs_events = p60 * p_cs_on
    t["clean_sheet"] = cs_events * scoring["cs"][pos]
    t["conceded"] = -p_start * expected_floor_div(opp_xg * ms, 2) if pos in (1, 2) else 0.0
    saves = r["saves"] * ms * opp_mult if pos == 1 else 0.0
    t["saves"] = p_start * expected_floor_div(saves, 3) if pos == 1 else 0.0

    thr = mc["_thr"].get(pos)
    p_dc = 0.0
    if thr and scoring["defcon"].get(pos, 0):
        disp = mc["_disp"]
        p_dc = (p_start * count_tail(r["dc"] * ms, thr, disp)
                + p_sub * count_tail(r["dc"] * sub_m / 90, thr, disp))
    t["defcon"] = p_dc * scoring["defcon"][pos]

    # Bonus: (a) season rate, (b) event model from this fixture's expected events.
    ev = mc["_bonus_event"][pos]
    b_event = (ev["goal"] * goals + ev["assist"] * assists + ev["cs"] * cs_events
               + ev["saves"] * p_start * saves + ev["defcon"] * p_dc + ev["p60"] * p60)
    w = mc["_bonus_blend"]
    t["bonus"] = w * r["bonus"] * xmins / 90 + (1 - w) * b_event
    return {"terms": t, "xmins": xmins, "p_start": p_start, "p60": p60}


def project(bootstrap: dict, fixtures: list[dict], start_gw: int, horizon: int, cfg: dict,
            team_model=None) -> pd.DataFrame:
    """`team_model` needs .expected_goals(team, opp, home), .p_clean_sheet(team, opp, home) and
    .league_avg (see model/team_strength.TeamModel). If None, FPL's strength_* fields are used,
    which only works when they are non-zero.

    Output: one row per player with xp_<gw> per GW and xp_total, plus GW+1 diagnostics
    (xmins, p_start, p60, model_xp = pure-model xPts before the ep_next blend) and xmins_<gw>."""
    mcfg = model_config(cfg)
    mc = dict(mcfg["minutes"])
    mc["_thr"] = {NAME_POS[n]: int(v) for n, v in mcfg["defcon"]["threshold"].items()}
    mc["_disp"] = _f(mcfg["defcon"]["dispersion"])
    mc["_bonus_event"] = {NAME_POS[n]: v for n, v in mcfg["bonus"]["event"].items()}
    mc["_bonus_blend"] = _f(mcfg["bonus"]["blend"])
    sc = mcfg["shrinkage"]
    scoring = scoring_from_bootstrap(bootstrap)
    w_ep = _f(mcfg.get("blend_fpl_ep"))

    tm = team_model if team_model is not None else _default_team_model(bootstrap, mcfg)
    teams = [t["id"] for t in bootstrap["teams"]]
    match_cache: dict[tuple[int, int, bool], tuple[float, float]] = {}

    def match(team: int, opp: int, home: bool) -> tuple[float, float]:
        key = (team, opp, home)
        if key not in match_cache:
            match_cache[key] = (tm.expected_goals(team, opp, home), tm.p_clean_sheet(team, opp, home))
        return match_cache[key]

    # Team's average xG for / against over all opponents and venues: the baseline the player's
    # season per-90 rates were (roughly) earned against.
    avg_for, avg_against = {}, {}
    for t in teams:
        f = [match(t, o, h)[0] for o in teams if o != t for h in (True, False)]
        a = [match(o, t, not h)[0] for o in teams if o != t for h in (True, False)]
        avg_for[t], avg_against[t] = sum(f) / len(f), sum(a) / len(a)

    played = team_games(fixtures, start_gw)
    fallback_games = max(1, start_gw - 1)
    deadlines = _deadlines(bootstrap)
    ref = deadlines.get(start_gw)
    priors = rate_priors(bootstrap["elements"], sc)

    by_team_gw: dict[tuple[int, int], list[tuple[int, bool]]] = {}
    for fx in fixtures:
        ev = fx.get("event")
        if ev is None or not (start_gw <= ev < start_gw + horizon):
            continue
        by_team_gw.setdefault((fx["team_h"], ev), []).append((fx["team_a"], True))
        by_team_gw.setdefault((fx["team_a"], ev), []).append((fx["team_h"], False))

    rows = []
    for p in bootstrap["elements"]:
        pos, team = p["element_type"], p["team"]
        games = _eligible_games(p, played.get(team, []), fallback_games)
        mp = minutes_profile(p, games, mc)
        r = shrunk_rates(p, priors, sc)
        row = {
            "id": p["id"], "name": p["web_name"], "team": team, "pos": pos,
            "cost": p["now_cost"] / 10, "status": p.get("status"), "news": p.get("news", ""),
            "selected_by": _f(p.get("selected_by_percent")),
        }
        for k, gw in enumerate(range(start_gw, start_gw + horizon)):
            avail = availability(p, k, deadlines.get(gw), ref, mc)
            xp = xmins = p_start = p60 = 0.0
            for opp_id, home in by_team_gw.get((team, gw), []):
                team_xg, p_cs = match(team, opp_id, home)
                opp_xg = match(opp_id, team, not home)[0]
                fp = fixture_points(pos, mp, avail, r, team_xg, opp_xg, p_cs,
                                    team_xg / avg_for[team], opp_xg / avg_against[team], scoring, mc)
                xp += sum(fp["terms"].values())
                xmins += fp["xmins"]
                p_start = 1 - (1 - p_start) * (1 - fp["p_start"])  # P(starts at least once)
                p60 += fp["p60"]
            if k == 0:
                row.update({"xmins": round(xmins, 1), "p_start": round(p_start, 3),
                            "p60": round(p60, 3), "model_xp": round(xp, 3)})
                if w_ep > 0 and p.get("ep_next") is not None:
                    xp = (1 - w_ep) * xp + w_ep * _f(p["ep_next"])
            row[f"xmins_{gw}"] = round(xmins, 1)
            row[f"xp_{gw}"] = round(xp, 3)
        rows.append(row)

    df = pd.DataFrame(rows)
    xp_cols = [c for c in df.columns if c.startswith("xp_") and c[3:].isdigit()]
    df["xp_total"] = df[xp_cols].sum(axis=1)
    return df
