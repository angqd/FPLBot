"""Deterministic expected-points (xPts) model.

For each player and each upcoming GW in the horizon:

  xPts = P(plays) * [ appearance + attack + clean_sheet + defcon + saves + bonus - conceded ]

  attack       = xG90 * goal_pts[pos] * atk_mult + xA90 * 3 * atk_mult           (scaled to expected minutes)
  clean_sheet  = P(CS) * cs_pts[pos],  P(CS) = exp(-opp_expected_goals)          (Poisson)
  defcon       = P(hit DC threshold) * 2                                          (2025/26+ rule)
  conceded     = -opp_expected_goals / 2   for GK/DEF
  opp_expected_goals / atk_mult / P(CS) come from a TeamModel (model/team_strength.py:
  time-decayed Poisson fit on results + xG). atk_mult = team_xg / league_avg.
  FPL's own strength_* fields are only used as a fallback when they're non-zero (they're 0 live).

Doubles sum both fixtures, blanks give 0. GW+1 is blended with FPL's own `ep_next`, which
already bakes in news/injury info the stat model can't see.

It's intentionally simple and transparent — every term is inspectable, and you can swap in a
learned model later (train on the vaastav/Fantasy-Premier-League history) behind the same
`project()` interface.
"""
from __future__ import annotations

import math

import pandas as pd

DEFAULT_SCORING = {
    "goal": {1: 10, 2: 6, 3: 5, 4: 4},
    "assist": 3,
    "cs": {1: 4, 2: 4, 3: 1, 4: 0},
    "defcon": 2,
    "defcon_threshold": {1: 99, 2: 10, 3: 12, 4: 12},
}


def _f(x, default=0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


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


def _p_plays(p: dict, gws_played: int) -> tuple[float, float]:
    """Return (P(appears), expected minutes | appears)."""
    if p.get("status") in ("i", "s", "u", "n"):
        chance = _f(p.get("chance_of_playing_next_round"), 0) / 100
    else:
        c = p.get("chance_of_playing_next_round")
        chance = 1.0 if c is None else _f(c) / 100
    mins = _f(p.get("minutes"))
    starts = _f(p.get("starts"), mins / 80)
    share = min(1.0, mins / max(1, gws_played * 90))
    start_rate = min(1.0, starts / max(1, gws_played))
    p_app = chance * min(1.0, 0.15 + 0.85 * max(share, start_rate))
    exp_min = 85 if start_rate > 0.7 else max(20, mins / max(1, starts or 1))
    return p_app, min(90, exp_min)


def project(bootstrap: dict, fixtures: list[dict], start_gw: int, horizon: int, cfg: dict,
            team_model=None) -> pd.DataFrame:
    """`team_model` needs .expected_goals(team, opp, home), .p_clean_sheet(team, opp, home) and
    .league_avg (see model/team_strength.TeamModel). If None, FPL's strength_* fields are used,
    which only works when they are non-zero."""
    scoring = DEFAULT_SCORING
    tm = team_model if team_model is not None else _default_team_model(bootstrap, cfg)
    league_avg = tm.league_avg
    match_cache: dict[tuple[int, int, bool], tuple[float, float]] = {}

    def match(team: int, opp: int, home: bool) -> tuple[float, float]:
        key = (team, opp, home)
        if key not in match_cache:
            match_cache[key] = (tm.expected_goals(team, opp, home), tm.p_clean_sheet(team, opp, home))
        return match_cache[key]
    gws_played = max(1, start_gw - 1)

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
        p_app, exp_min = _p_plays(p, gws_played)
        reliable = _f(p.get("minutes")) >= cfg["min_minutes_for_rates"]
        shrink = 1.0 if reliable else 0.5  # small samples -> regress per-90 rates
        xg90 = _f(p.get("expected_goals_per_90")) * shrink
        xa90 = _f(p.get("expected_assists_per_90")) * shrink
        dc90 = _f(p.get("defensive_contribution_per_90"))
        sv90 = _f(p.get("saves_per_90"))
        bonus_pg = _f(p.get("bonus")) / gws_played
        m = exp_min / 90

        row = {
            "id": p["id"], "name": p["web_name"], "team": team, "pos": pos,
            "cost": p["now_cost"] / 10, "status": p.get("status"), "news": p.get("news", ""),
            "selected_by": _f(p.get("selected_by_percent")),
        }
        for gw in range(start_gw, start_gw + horizon):
            xp = 0.0
            for opp_id, home in by_team_gw.get((team, gw), []):
                team_xg, p_cs = match(team, opp_id, home)
                opp_xg = match(opp_id, team, not home)[0]
                atk_mult = team_xg / league_avg
                pts = 2 if exp_min >= 60 else 1
                pts += (xg90 * scoring["goal"][pos] + xa90 * scoring["assist"]) * m * atk_mult
                if exp_min >= 60:
                    pts += p_cs * scoring["cs"][pos]
                thr = scoring["defcon_threshold"][pos]
                if dc90 and thr < 99:
                    pts += scoring["defcon"] / (1 + math.exp(-1.2 * (dc90 * m - thr)))
                if pos == 1:
                    pts += sv90 * m / 3
                if pos in (1, 2):
                    pts -= opp_xg * m / 2
                pts += bonus_pg * 0.8
                xp += p_app * pts
            if gw == start_gw and p.get("ep_next") is not None:
                w = cfg["blend_fpl_ep"]
                xp = (1 - w) * xp + w * _f(p["ep_next"])
            row[f"xp_{gw}"] = round(xp, 3)
        rows.append(row)

    df = pd.DataFrame(rows)
    xp_cols = [c for c in df.columns if c.startswith("xp_")]
    df["xp_total"] = df[xp_cols].sum(axis=1)
    return df
