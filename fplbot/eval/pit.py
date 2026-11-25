"""Point-in-time reconstruction: what bootstrap-static / fixtures / results looked like at GW g's
deadline, built ONLY from GWs < g.

From GW g itself we read only what is public before the deadline: the player's price (`value`),
club and position, and which fixtures are scheduled in g. Every stat is a cumulative sum over
GWs < g. ep_next is the frozen baseline for g (see history.py).
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from fplbot.eval.history import STAT_COLS, SeasonData

# bootstrap-static *_per_90 fields: name -> cumulative stat it is derived from (per 90 minutes)
PER_90 = {
    "expected_goals_per_90": "expected_goals",
    "expected_assists_per_90": "expected_assists",
    "expected_goal_involvements_per_90": "expected_goal_involvements",
    "expected_goals_conceded_per_90": "expected_goals_conceded",
    "saves_per_90": "saves",
    "goals_conceded_per_90": "goals_conceded",
    "starts_per_90": "starts",
    "clean_sheets_per_90": "clean_sheets",
    "defensive_contribution_per_90": "defensive_contribution",
}
STRING_STATS = {"expected_goals", "expected_assists", "expected_goal_involvements", "expected_goals_conceded"}
ELEMENT_TYPES = [
    {"id": 1, "singular_name_short": "GKP", "squad_select": 2},
    {"id": 2, "singular_name_short": "DEF", "squad_select": 5},
    {"id": 3, "singular_name_short": "MID", "squad_select": 5},
    {"id": 4, "singular_name_short": "FWD", "squad_select": 3},
]


def _known_at(sd: SeasonData, g: int) -> pd.DataFrame:
    """Per element: club, position, price, name at GW g (from g's row if any, else last before g)."""
    rows = sd.gw[sd.gw["gw"] <= g].sort_values("gw")
    return rows.groupby("element")[["team", "element_type", "value", "web_name"]].last()


def bootstrap_at(sd: SeasonData, g: int, use_availability: bool = True) -> dict:
    """bootstrap-static-like dict for target GW g. Field names match the live API.

    status is 'a' unless `use_availability` and sd.avail has the player's chance of playing at
    g's deadline (recovered from FPL's frozen chance_of_playing_this_round, see history.py):
    chance 0 -> status 'i', 0 < chance < 1 -> 'd', with chance_of_playing_next_round set."""
    past = sd.gw[sd.gw["gw"] < g]
    known = _known_at(sd, g)
    cum = past.groupby("element")[STAT_COLS].sum()
    apps = past[past["minutes"] > 0].groupby("element").size()
    pts_by_gw = past.pivot_table(index="element", columns="gw", values="total_points", aggfunc="sum")
    last4 = [c for c in pts_by_gw.columns if c >= g - 4]
    form = pts_by_gw[last4].fillna(0).sum(axis=1) / max(1, len(last4)) if last4 else pd.Series(dtype=float)
    last_pts = pts_by_gw[g - 1] if (g - 1) in pts_by_gw.columns else pd.Series(dtype=float)
    ep = sd.ep[sd.ep["gw"] == g].set_index("element")["ep"]
    avail = sd.avail[sd.avail["gw"] == g].set_index("element")["chance"] if use_availability else pd.Series(dtype=float)

    elements = []
    for pid, k in known.iterrows():
        c = cum.loc[pid] if pid in cum.index else pd.Series(0.0, index=STAT_COLS)
        mins = float(c["minutes"])
        e = {"id": int(pid), "web_name": str(k["web_name"]), "team": int(k["team"]),
             "element_type": int(k["element_type"]), "now_cost": int(k["value"]),
             "status": "a", "chance_of_playing_next_round": None, "chance_of_playing_this_round": None,
             "news": "", "selected_by_percent": "0.0"}
        for s in STAT_COLS:
            e[s] = f"{c[s]:.2f}" if s in STRING_STATS else int(round(c[s]))
        for f, s in PER_90.items():
            e[f] = round(float(c[s]) / mins * 90, 2) if mins > 0 else 0.0
        n_app = int(apps.get(pid, 0))
        e["points_per_game"] = f"{c['total_points'] / n_app:.1f}" if n_app else "0.0"
        e["form"] = f"{float(form.get(pid, 0.0)):.1f}"
        e["event_points"] = int(last_pts.get(pid, 0) if pd.notna(last_pts.get(pid, 0)) else 0)
        e["ep_next"] = f"{float(ep[pid]):.1f}" if pid in ep.index else None
        e["ep_this"] = None
        ch = avail.get(pid)
        if ch is not None and float(ch) < 1:
            e["status"] = "i" if float(ch) == 0 else "d"
            e["chance_of_playing_next_round"] = e["chance_of_playing_this_round"] = int(round(float(ch) * 100))
        elements.append(e)

    events = []
    for i in sorted(set(sd.deadlines) | {g}):
        events.append({"id": i, "name": f"Gameweek {i}", "deadline_time": sd.deadlines.get(i),
                       "finished": i < g, "data_checked": i < g, "is_previous": i == g - 1,
                       "is_current": i == g - 1, "is_next": i == g})
    return {"elements": elements, "teams": [dict(t) for t in sd.teams], "events": events,
            "element_types": ELEMENT_TYPES}


def fixtures_at(sd: SeasonData, g: int) -> list[dict]:
    """Fixture list as of g's deadline: schedule only (no scores), finished flag for events < g.
    Note: uses the final event assignment, so a fixture postponed after g's deadline already
    sits in its rescheduled GW (minor lookahead, mostly relevant for horizon > 1)."""
    return [{**{k: fx[k] for k in ("id", "event", "team_h", "team_a", "kickoff_time")},
             "finished": fx["event"] is not None and fx["event"] < g}
            for fx in sd.fixtures]


def deadline(sd: SeasonData, g: int) -> datetime:
    return datetime.fromisoformat(sd.deadlines[g].replace("Z", "+00:00"))


def results_before(results: pd.DataFrame, when: datetime) -> pd.DataFrame:
    """Matches played strictly before the deadline's calendar date (football-data has dates only,
    and matches on deadline day kick off after it)."""
    cut = pd.Timestamp(when.date())
    return results[pd.to_datetime(results["date"]) < cut].reset_index(drop=True)
