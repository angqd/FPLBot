"""Backtest harness tests, offline on a small synthetic season written in vaastav's CSV layout."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from fplbot.config import load_config
from fplbot.eval import backtest as bt
from fplbot.eval import pit
from fplbot.eval.history import (aggregate_fixture_rows, form_at, live_to_rows, load_season,
                                 snapshot_eps, _points_wide)
from fplbot.model.xpts import project

CFG = load_config().raw
SEASON = "2030-31"
N_TEAMS, N_GW = 6, 8
START = datetime(2030, 8, 17, 14, tzinfo=timezone.utc)
POS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def _round_robin(n):
    ids = list(range(1, n + 1))
    rounds = []
    for _ in range(n - 1):
        rounds.append([(ids[i], ids[-1 - i]) for i in range(n // 2)])
        ids = [ids[0]] + [ids[-1]] + ids[1:-1]
    return rounds + [[(a, h) for h, a in r] for r in rounds]


def make_season(seed=0):
    """-> (merged_gw, teams, fixtures, results) DataFrames for a synthetic season."""
    rng = np.random.default_rng(seed)
    teams = pd.DataFrame({"id": range(1, N_TEAMS + 1), "name": [f"Club {i}" for i in range(1, N_TEAMS + 1)],
                          "short_name": [f"C{i}" for i in range(1, N_TEAMS + 1)], "code": range(1, N_TEAMS + 1),
                          "strength_overall_home": 3, "strength_overall_away": 3})
    q = rng.normal(0, 0.3, N_TEAMS)
    fx, res, rows, fid = [], [], [], 1
    shape = [1, 2, 2, 3, 3, 4]
    players = [(i * len(shape) + k + 1, t, pos) for i, t in enumerate(range(1, N_TEAMS + 1))
               for k, pos in enumerate(shape)]
    skill = {p: rng.normal(0, 1) for p, _, _ in players}
    rounds = _round_robin(N_TEAMS)
    for g, rnd in enumerate(rounds[:N_GW], start=1):
        ko = START + timedelta(days=7 * (g - 1))
        for h, a in rnd:
            lh, la = np.exp(0.3 + 0.2 + q[h - 1] - q[a - 1]), np.exp(0.3 + q[a - 1] - q[h - 1])
            hg, ag = int(rng.poisson(lh)), int(rng.poisson(la))
            fx.append({"id": fid, "event": g, "kickoff_time": ko.strftime("%Y-%m-%dT%H:%M:%SZ"),
                       "team_h": h, "team_a": a, "team_h_score": hg, "team_a_score": ag, "finished": True})
            res.append({"Date": ko.strftime("%d/%m/%Y"), "HomeTeam": f"Club {h}", "AwayTeam": f"Club {a}",
                        "FTHG": hg, "FTAG": ag, "HxG": round(lh, 2), "AxG": round(la, 2)})
            for pid, t, pos in players:
                if t not in (h, a):
                    continue
                mins = int(rng.choice([0, 90, 90, 90, 60, 20]))
                xg = max(0.0, 0.1 * pos * (1 + 0.3 * skill[pid]) * mins / 90)
                pts = (0 if mins == 0 else 2) + int(rng.poisson(xg) * 5)
                rows.append({"name": f"Player {pid}", "position": POS[pos], "team": f"Club {t}",
                             "xP": round(rng.uniform(0, 6), 1), "element": pid, "fixture": fid,
                             "kickoff_time": ko.strftime("%Y-%m-%dT%H:%M:%SZ"), "minutes": mins,
                             "starts": int(mins >= 60), "goals_scored": 0, "assists": 0, "clean_sheets": 0,
                             "goals_conceded": 0, "saves": 2 if pos == 1 else 0, "bonus": int(pts > 6),
                             "bps": 10, "defensive_contribution": int(rng.integers(0, 12)),
                             "expected_goals": round(xg, 2), "expected_assists": 0.05,
                             "expected_goal_involvements": round(xg + 0.05, 2), "expected_goals_conceded": 1.0,
                             "total_points": pts, "value": 50 + 5 * pos, "round": g, "GW": g,
                             "was_home": t == h, "opponent_team": a if t == h else h})
        fid = len(fx) + 1
    # one more scheduled but unplayed GW so horizon>1 has fixtures
    for h, a in rounds[N_GW]:
        ko = START + timedelta(days=7 * N_GW)
        fx.append({"id": len(fx) + 1, "event": N_GW + 1, "kickoff_time": ko.strftime("%Y-%m-%dT%H:%M:%SZ"),
                   "team_h": h, "team_a": a, "team_h_score": None, "team_a_score": None, "finished": False})
    return pd.DataFrame(rows), teams, pd.DataFrame(fx), pd.DataFrame(res)


@pytest.fixture()
def season(tmp_path):
    merged, teams, fx, res = make_season()
    d = tmp_path / "cache" / "vaastav" / SEASON
    d.mkdir(parents=True)
    merged.to_csv(d / "gws_merged_gw.csv", index=False)
    teams.to_csv(d / "teams.csv", index=False)
    fx.to_csv(d / "fixtures.csv", index=False)
    sd = load_season(SEASON, {"current_season": None}, offline=True, data_dir=tmp_path)
    from fplbot.model.team_strength import normalize_results
    return sd, normalize_results(res)


def _project_gw(sd, results, g, horizon=2, model_cfg=None):
    bs = pit.bootstrap_at(sd, g)
    fx = pit.fixtures_at(sd, g)
    from fplbot.model.team_strength import fit_team_model
    tm = fit_team_model(bs, CFG, results=pit.results_before(results, pit.deadline(sd, g)))
    return project(bs, fx, g, horizon, model_cfg or CFG["model"], team_model=tm)


def test_load_season_aggregates_and_deadlines(season):
    sd, _ = season
    assert sd.finished_gws == list(range(1, N_GW + 1))
    assert len(sd.teams) == N_TEAMS and all(t["strength_attack_home"] == 0 for t in sd.teams)
    # deadline = first kickoff - 90 min
    assert sd.deadlines[1] == (START - timedelta(minutes=90)).strftime("%Y-%m-%dT%H:%M:%SZ")
    assert set(sd.ep["source"]) == {"reconstructed_form"}
    assert set(sd.posthoc["source"]) == {"vaastav_xP_posthoc"}


def test_double_gw_rows_are_summed():
    rows = pd.DataFrame([
        {"name": "A", "position": "MID", "team": "X", "xP": 3.0, "element": 1, "fixture": f, "GW": 5,
         "kickoff_time": k, "minutes": 90, "total_points": p, "value": 60, "expected_goals": 0.5}
        for f, k, p in ((10, "2030-01-01T12:00:00Z", 2), (11, "2030-01-04T12:00:00Z", 7))])
    gw, ph = aggregate_fixture_rows(rows, pd.DataFrame({"id": [3], "name": ["X"]}))
    r = gw.iloc[0]
    assert (r.minutes, r.total_points, r.n_fixtures, r.team, r.value) == (180, 9, 2, 3, 60)
    assert r.expected_goals == pytest.approx(1.0)
    assert ph.iloc[0].ep == 3.0


def test_bootstrap_at_field_names_and_values(season):
    sd, _ = season
    g = 5
    bs = pit.bootstrap_at(sd, g)
    real_fields = {"id", "web_name", "team", "element_type", "now_cost", "status", "chance_of_playing_next_round",
                   "minutes", "starts", "goals_scored", "assists", "clean_sheets", "saves", "bonus",
                   "defensive_contribution", "expected_goals", "expected_assists", "total_points",
                   "expected_goals_per_90", "expected_assists_per_90", "defensive_contribution_per_90",
                   "saves_per_90", "starts_per_90", "clean_sheets_per_90", "ep_next", "form", "points_per_game"}
    e = bs["elements"][0]
    assert real_fields <= set(e)
    past = sd.gw[(sd.gw.gw < g) & (sd.gw.element == e["id"])]
    assert e["minutes"] == past.minutes.sum()
    assert e["total_points"] == past.total_points.sum()
    assert isinstance(e["expected_goals"], str)  # strings in the live API
    if e["minutes"]:
        assert e["saves_per_90"] == round(past.saves.sum() / past.minutes.sum() * 90, 2)
    nxt = [ev for ev in bs["events"] if ev["is_next"]]
    assert [ev["id"] for ev in nxt] == [g]
    assert all(ev["finished"] == (ev["id"] < g) for ev in bs["events"])
    assert e["ep_next"] is not None


def test_no_lookahead(season):
    """Mutating anything from GW g onwards (stats, points, xP, results after the deadline) must not
    change projections for g, nor the ep / naive baselines."""
    sd, results = season
    g = 5
    before = _project_gw(sd, results, g)
    pred_before = bt.predict_gw(sd, g, 2, CFG, results, bt.DEFAULT_VARIANTS)

    future = sd.gw["gw"] >= g
    rng = np.random.default_rng(99)
    mutated = sd.gw.copy()
    for c in ("minutes", "starts", "total_points", "expected_goals", "expected_assists", "bonus", "saves",
              "defensive_contribution", "goals_scored"):
        mutated.loc[future, c] = rng.integers(0, 50, future.sum())
    sd.gw = mutated
    ph = sd.posthoc.copy()
    ph.loc[ph["gw"] >= g, "ep"] = 99.0  # leaky post-hoc ep must not leak in (only its chance ratio)
    sd.posthoc = ph
    from fplbot.eval.history import reconstruct_baseline
    sd.ep, sd.avail = reconstruct_baseline(sd.gw, sd.deadlines, sd.fixtures, sd.posthoc)
    dl = pit.deadline(sd, g)
    res2 = results.copy()
    late = pd.to_datetime(res2["date"]) >= pd.Timestamp(dl.date())
    res2.loc[late, ["hg", "ag", "hxg", "axg"]] = 9.0

    after = _project_gw(sd, res2, g)
    pd.testing.assert_frame_equal(before, after)
    pred_after = bt.predict_gw(sd, g, 2, CFG, res2, bt.DEFAULT_VARIANTS)
    pd.testing.assert_frame_equal(pred_before.drop(columns="ep_posthoc"), pred_after.drop(columns="ep_posthoc"))


def test_lookahead_would_be_detected(season):
    """Sanity check for the test above: changing a PAST GW does change projections."""
    sd, results = season
    before = _project_gw(sd, results, 5)
    sd.gw.loc[sd.gw["gw"] == 3, "expected_goals"] += 2.0
    assert not before.equals(_project_gw(sd, results, 5))


def test_form_baseline_matches_definition(season):
    sd, _ = season
    g = 6
    pts = _points_wide(sd.gw)
    f = form_at(pts, sd.deadlines, g)
    # weekly GWs: the 30-day window before GW6's deadline holds GWs 2..5
    assert f.equals((pts[[2, 3, 4, 5]].sum(axis=1) / 4).round(1))
    ep = sd.ep[sd.ep.gw == g].set_index("element")["ep"]
    assert (ep <= f.reindex(ep.index) + 1e-9).all()


def test_availability_recovered_from_posthoc():
    """ep_this(g) = form-including-g x chance  ->  chance (quarters), 0 for flagged players."""
    from fplbot.eval.history import availability_from_posthoc
    pts = pd.DataFrame({1: [4, 2, 6, 0], 2: [2, 2, 6, 3], 3: [6, 2, 0, 3]}, index=[10, 11, 12, 13], dtype=float)
    form_incl = pts.mean(axis=1).round(1)
    ep_this = form_incl * pd.Series([1.0, 0.5, 0.0, 1.0], index=pts.index)
    chance, rate = availability_from_posthoc(pts, ep_this, 3)
    assert chance.to_dict() == {10: 1.0, 11: 0.5, 12: 0.0, 13: 1.0}
    assert rate == pytest.approx(0.5)


def test_snapshot_eps_pre_and_post_deadline():
    dl = "2030-09-01T10:00:00Z"
    ev = lambda nxt, cur: [{"id": 4, "deadline_time": dl, "is_next": nxt, "is_current": cur}]
    el = lambda v: [{"id": 1, "ep_next": str(v), "ep_this": str(v + 1)}]
    snaps = [(datetime(2030, 8, 30, tzinfo=timezone.utc), {"events": ev(True, False), "elements": el(3.0)}),
             (datetime(2030, 9, 1, 9, tzinfo=timezone.utc), {"events": ev(True, False), "elements": el(4.0)}),
             (datetime(2030, 9, 2, tzinfo=timezone.utc), {"events": ev(False, True), "elements": el(7.0)})]
    pre, post = snapshot_eps(snaps)
    assert pre.to_dict("records") == [{"element": 1, "gw": 4, "ep": 4.0, "source": "snapshot_ep_next_pre_deadline"}]
    assert post.iloc[0].ep == 8.0 and post.iloc[0].source == "snapshot_ep_this_posthoc"


def test_live_to_rows():
    live = {"elements": [{"id": 7, "stats": {"minutes": 90, "total_points": 6, "expected_goals": "0.40"}}]}
    elements = [{"id": 7, "team": 2, "element_type": 3, "web_name": "X", "now_cost": 65}]
    df = live_to_rows(live, 11, elements, [{"event": 11, "team_h": 2, "team_a": 5}])
    r = df.iloc[0]
    assert (r.gw, r.team, r.minutes, r.total_points, r.n_fixtures, r.value) == (11, 2, 90, 6, 1, 65)
    assert r.expected_goals == pytest.approx(0.4)


def test_run_backtest_and_tuning_hook(season, tmp_path):
    sd, results = season
    r = bt.run_backtest([sd], CFG, {SEASON: results}, horizons=[1, 3])
    s = r.summary
    assert {"model_pure", "model_blend", "ep", "naive", "ep_posthoc"} <= set(s["model"])
    h1 = s[(s.scope == "all_gws") & (s.horizon == 1) & (s.set == "decision")]
    assert np.isfinite(h1[["mae", "rmse", "captain_pts", "xi_pts"]].to_numpy(float)).all()
    assert set(r.per_gw["gw"]) == set(range(2, N_GW + 1))
    assert r.per_gw[r.per_gw.horizon == 3]["gw"].max() == N_GW - 2
    # blend 0 is the pure model
    out = bt.evaluate_configs([{"model": {"blend_fpl_ep": 0.0}}, {"model": {"blend_fpl_ep": 1.0}}],
                              [sd], CFG, {SEASON: results}, horizons=(1,))
    assert len(out) == 2
    pure = h1.set_index("model").loc["model_pure", "mae"]
    assert out[0]["metrics"][1]["mae"] == pytest.approx(pure)
    ep_mae = s[(s.scope == "all_gws") & (s.horizon == 1) & (s.set == "decision")].set_index("model").loc["ep", "mae"]
    # blend 1.0 = ep exactly (decision sets differ, so compare on 'played')
    out_p = bt.evaluate_configs([{"model": {"blend_fpl_ep": 1.0}}], [sd], CFG, {SEASON: results},
                                horizons=(1,), set_name="played")
    ep_played = s[(s.scope == "all_gws") & (s.horizon == 1) & (s.set == "played")].set_index("model").loc["ep", "mae"]
    assert out_p[0]["metrics"][1]["mae"] == pytest.approx(ep_played, abs=1e-3)
    assert ep_mae > 0
    paths = bt.save(r, tmp_path / "eval")
    assert paths["summary_csv"].exists()


def test_best_xi_formation():
    pred = pd.Series({i: float(i) for i in range(1, 21)})
    pos = pd.Series({i: (1 if i <= 2 else 2 if i <= 8 else 3 if i <= 14 else 4) for i in range(1, 21)})
    xi = bt.best_xi(pred, pos)
    counts = pos[xi].value_counts().to_dict()
    assert len(xi) == 11 and counts[1] == 1 and 3 <= counts[2] <= 5 and 2 <= counts[3] <= 5 and 1 <= counts[4] <= 3
