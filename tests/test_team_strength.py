import math

import numpy as np
import pandas as pd
import pytest

from fplbot.config import load_config
from fplbot.model.team_strength import (TeamMappingError, TeamModel, fit_ratings, load_results,
                                        map_teams, parse_results_csv)
from fplbot.model.xpts import project
from tests.make_fixture import NEXT_GW, make, make_results, simulate_results

CFG = load_config().raw
NAMES = [f"T{i:02d}" for i in range(20)]


def _ranks(x):
    return np.argsort(np.argsort(x))


def _csv_roundtrip(df):
    return parse_results_csv(df.to_csv(index=False))


def test_fit_recovers_known_ratings():
    rng = np.random.default_rng(3)
    att = rng.normal(0, 0.25, 20)
    dfn = rng.normal(0, 0.25, 20)
    att, dfn = att - att.mean(), dfn - dfn.mean()
    res = simulate_results(dict(zip(NAMES, att)), dict(zip(NAMES, dfn)), mu=0.2, home=0.3, seasons=3)
    r = fit_ratings(_csv_roundtrip(res), {"xi": 0.0, "ridge_sigma": 1.0})

    fit_a = np.array([r.att[n] for n in NAMES])
    fit_d = np.array([r.dfn[n] for n in NAMES])
    assert abs(fit_a.sum()) < 1e-6 and abs(fit_d.sum()) < 1e-6
    assert np.corrcoef(_ranks(att), _ranks(fit_a))[0, 1] > 0.9
    assert np.corrcoef(_ranks(dfn), _ranks(fit_d))[0, 1] > 0.9
    assert np.max(np.abs(fit_a - att)) < 0.12 and np.max(np.abs(fit_d - dfn)) < 0.12
    assert r.home == pytest.approx(0.3, abs=0.06) and r.mu == pytest.approx(0.2, abs=0.06)
    # deterministic
    r2 = fit_ratings(_csv_roundtrip(res), {"xi": 0.0, "ridge_sigma": 1.0})
    assert r2.att == r.att


def test_ridge_shrinks_and_decay_downweights_old_matches():
    att = {n: (0.4 if n == "T00" else -0.4 / 19) for n in NAMES}
    flat = {n: 0.0 for n in NAMES}
    old = _csv_roundtrip(simulate_results(att, flat, seasons=1))
    loose = fit_ratings(old, {"xi": 0.0, "ridge_sigma": 1.0})
    tight = fit_ratings(old, {"xi": 0.0, "ridge_sigma": 0.05})
    assert loose.att["T00"] > tight.att["T00"] > 0
    # T00 was strong only last season; with heavier decay the recent (average) season dominates
    new = _csv_roundtrip(simulate_results(flat, flat, seasons=1, seed=5,
                                          start=(old.date.max() + pd.Timedelta(days=80)).date()))
    both = pd.concat([old, new], ignore_index=True)
    assert fit_ratings(both, {"xi": 0.01}).att["T00"] < fit_ratings(both, {"xi": 0.0}).att["T00"]


def test_zero_strength_bootstrap_projects_with_team_model():
    bs, fx = make()
    assert all(t["strength_attack_home"] == 0 for t in bs["teams"])
    with pytest.raises(ValueError, match="team_model"):
        project(bs, fx, NEXT_GW, 2, CFG["model"])
    tm = TeamModel.from_results(make_results(), bs, CFG)
    proj = project(bs, fx, NEXT_GW, 2, CFG["model"], team_model=tm)
    assert proj[[f"xp_{NEXT_GW}", f"xp_{NEXT_GW + 1}"]].notna().all().all()
    assert proj.xp_total.max() > 5
    for t in bs["teams"]:
        for o in bs["teams"][:3]:
            if o["id"] != t["id"]:
                p = tm.p_clean_sheet(t["id"], o["id"], True)
                assert 0 < p < 1
                assert p == pytest.approx(math.exp(-tm.expected_goals(o["id"], t["id"], False)))
    assert 1.0 < tm.league_avg < 2.0


def test_mapping_uses_aliases_and_ignores_extra_result_teams():
    bs = {"teams": [{"id": 1, "name": "Man Utd", "short_name": "MUN"},
                    {"id": 2, "name": "Spurs", "short_name": "TOT"},
                    {"id": 3, "name": "Nott'm Forest", "short_name": "NFO"},
                    {"id": 4, "name": "Coventry City", "short_name": "COV"}]}
    m = map_teams(bs, ["Man United", "Tottenham", "Nott'm Forest", "Coventry", "Burnley"])
    assert m == {1: "Man United", 2: "Tottenham", 3: "Nott'm Forest", 4: "Coventry"}


def test_mapping_raises_on_unknown_team():
    bs, _ = make()
    bs["teams"][4]["name"], bs["teams"][4]["short_name"] = "Atlantis FC", "ATL"
    with pytest.raises(TeamMappingError, match="Atlantis FC"):
        TeamModel.from_results(make_results(), bs, CFG)


def test_offline_cache_and_missing_xg(tmp_path):
    csv = make_results().drop(columns=["HxG", "AxG"]).to_csv(index=False)
    (tmp_path / "football-data_E0_2627.csv").write_text("﻿" + csv)
    res = load_results({"seasons": ["2627"]}, offline=True, cache_dir=tmp_path)
    assert res.hxg.isna().all() and len(res) > 100
    r = fit_ratings(res, CFG)  # falls back to goals only
    assert all(math.isfinite(v) for v in r.att.values())
    with pytest.raises(FileNotFoundError):
        load_results({"seasons": ["2526"]}, offline=True, cache_dir=tmp_path)
