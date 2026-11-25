import logging
import math

import numpy as np
import pandas as pd
import pytest

from fplbot.model import odds_api
from fplbot.model.odds import devig, implied_lambdas, outcome_probs
from fplbot.model.team_strength import (_cfg, build_target, detect_promoted, fit_ratings,
                                        fit_team_model, normalize_results, parse_results_csv)
from tests.make_fixture import simulate_results

OLD = [f"T{i:02d}" for i in range(20)]
NEW = OLD[:17] + ["P1", "P2", "P3"]  # T17..T19 relegated, P1..P3 promoted
FLAT_OLD = {n: 0.0 for n in OLD}
FLAT_NEW = {n: 0.0 for n in NEW}
CFG = {"xi": 0.0, "ridge_sigma": 0.2, "promoted_prior": {"att": -0.28, "def": -0.21}}


def _two_seasons(current_rounds, new_att=None, seed=1):
    last = simulate_results(FLAT_OLD, FLAT_OLD, seasons=0, current_rounds=38, start=pd.Timestamp(2025, 8, 16).date())
    if current_rounds:
        cur = simulate_results(new_att or FLAT_NEW, FLAT_NEW, seasons=0, current_rounds=current_rounds,
                               seed=seed, start=pd.Timestamp(2026, 8, 15).date())
        last = pd.concat([last, cur])
    return parse_results_csv(last.to_csv(index=False))


def test_detect_promoted_uses_only_the_frame():
    res = _two_seasons(3)
    assert detect_promoted(res) == {"P1", "P2", "P3"}
    # previous season absent from the frame -> nothing detected (no peeking at other data)
    assert detect_promoted(res[res.season == 2026]) == set()
    # a frame cut before the new season started: latest season is 2025, nothing promoted
    assert detect_promoted(res[res.date < "2026-08-01"]) == set()


def test_prior_pulls_low_data_promoted_team_below_average_and_fades():
    early = _two_seasons(3)  # 3 matches each: P1..P3 are truly average
    with_p = fit_ratings(early, CFG)
    without = fit_ratings(early, {**CFG, "promoted_prior": {"enabled": False}})
    assert with_p.promoted == {"P1", "P2", "P3"}
    for t in ("P1", "P2", "P3"):
        assert with_p.att[t] < without.att[t] - 0.1 and with_p.att[t] < -0.1
        assert with_p.dfn[t] < without.dfn[t] - 0.07 and with_p.dfn[t] < -0.05
    # centred on this season's clubs
    assert abs(np.mean([with_p.att[t] for t in NEW])) < 1e-9

    # a full season of data: the prior's pull shrinks (ridge lets the data dominate)
    late = _two_seasons(38)
    late_p = fit_ratings(late, CFG)
    late_0 = fit_ratings(late, {**CFG, "promoted_prior": {"enabled": False}})
    pull_early = np.mean([without.att[t] - with_p.att[t] for t in ("P1", "P2", "P3")])
    pull_late = np.mean([late_0.att[t] - late_p.att[t] for t in ("P1", "P2", "P3")])
    assert 0 < pull_late < 0.6 * pull_early


def test_implied_lambdas_hand_example():
    # λ = (1.5, 1.0): total 2.5 -> P(≤2 goals) = e^-2.5 (1 + 2.5 + 3.125) = 0.543813
    p_h, p_d, p_a, p_o = outcome_probs([1.5], [1.0])
    assert p_o[0] == pytest.approx(1 - math.exp(-2.5) * 6.625, abs=1e-9)
    assert p_h[0] + p_d[0] + p_a[0] == pytest.approx(1, abs=1e-9)
    # price with a 6% margin, write as a football-data CSV (Bet365 only: exercises the fallback)
    m = 1.06
    row = {"Date": "20/09/2026", "HomeTeam": "A", "AwayTeam": "B", "FTHG": 2, "FTAG": 1,
           "B365H": 1 / (p_h[0] * m), "B365D": 1 / (p_d[0] * m), "B365A": 1 / (p_a[0] * m),
           "B365>2.5": 1 / (p_o[0] * m), "B365<2.5": 1 / ((1 - p_o[0]) * m)}
    res = parse_results_csv(pd.DataFrame([row]).to_csv(index=False))
    assert res.hmkt[0] == pytest.approx(1.5, abs=1e-4) and res.amkt[0] == pytest.approx(1.0, abs=1e-4)
    # de-vig is proportional: 2.0/3.5/4.0 -> 1/odds normalised
    p = devig(np.array([[2.0, 3.5, 4.0]]))[0]
    assert p.sum() == pytest.approx(1) and p[0] == pytest.approx(0.5 / (0.5 + 1 / 3.5 + 0.25))
    # missing odds -> NaN, and closing averages win over Bet365 when both exist
    lh, _ = implied_lambdas([np.nan], [0.3], [0.5])
    assert np.isnan(lh[0])
    row2 = {**row, "AvgCH": 1.5, "AvgCD": 4.5, "AvgCA": 6.5, "AvgC>2.5": 1.6, "AvgC<2.5": 2.4}
    res2 = parse_results_csv(pd.DataFrame([row2]).to_csv(index=False))
    assert res2.hmkt[0] > 1.8  # big home favourite per the closing prices


def test_target_weights_renormalise_when_a_source_is_missing():
    df = pd.DataFrame({"hg": [2.0, 2.0, 2.0], "ag": [0.0, 0.0, 0.0],
                       "hxg": [1.0, np.nan, np.nan], "axg": [1.0, np.nan, np.nan],
                       "hmkt": [1.5, 1.5, np.nan], "amkt": [0.5, 0.5, np.nan]})
    yh, ya = build_target(df, {"goals": 0.3, "xg": 0.35, "odds": 0.35})
    assert yh[0] == pytest.approx(0.3 * 2 + 0.35 * 1 + 0.35 * 1.5)
    assert yh[1] == pytest.approx((0.3 * 2 + 0.35 * 1.5) / 0.65)  # no xG: goals 46%, odds 54%
    assert ya[1] == pytest.approx((0.35 * 0.5) / 0.65)
    assert yh[2] == 2.0 and ya[2] == 0.0  # goals only
    # legacy config still works and means no odds
    assert _cfg({"goals_weight": 0.5})["target_weights"] == {"goals": 0.5, "xg": 0.5, "odds": 0.0}
    assert _cfg(None)["target_weights"]["odds"] > 0


def test_odds_target_moves_ratings_toward_market():
    # results say T00 is average, the market says it is strong: odds weight pulls att up
    att = {n: (0.4 if n == "T00" else 0.0) for n in OLD}
    mkt = simulate_results(att, FLAT_OLD, seasons=1, seed=3)[["AvgH", "AvgD", "AvgA", "Avg>2.5", "Avg<2.5"]]
    flat = simulate_results(FLAT_OLD, FLAT_OLD, seasons=1, seed=3, odds=False)
    df = pd.concat([flat, mkt], axis=1)
    res = parse_results_csv(df.to_csv(index=False))
    loose = {**CFG, "ridge_sigma": 5.0}
    a0 = fit_ratings(res, {**loose, "target_weights": {"goals": 1, "xg": 0, "odds": 0}}).att["T00"]
    a1 = fit_ratings(res, {**loose, "target_weights": {"goals": 0, "xg": 0, "odds": 1}}).att["T00"]
    # noise-free market targets recover the truth (0.4 centred over 20 clubs = 0.38)
    assert a1 > a0 + 0.2 and a1 == pytest.approx(0.38, abs=0.02)


def test_unseen_teams_get_prior_and_explicit_results_never_fetch(monkeypatch):
    import httpx
    monkeypatch.setattr(httpx, "get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")))
    monkeypatch.setenv("ODDS_API_KEY", "sekrit")
    res = _two_seasons(0)  # new season not started: P1..P3 have no rows
    bs = {"teams": [{"id": i + 1, "name": n, "short_name": n} for i, n in enumerate(NEW)]}
    tm = fit_team_model(bs, CFG, results=res, unseen_as_promoted=True)
    assert tm.market == {}
    p1 = NEW.index("P1") + 1
    assert tm.att(p1) == pytest.approx(-0.28, abs=0.05) and tm.dfn(p1) < -0.1
    assert tm.expected_goals(p1, 1, True) < tm.league_avg * math.exp(tm.ratings.home / 2)


def test_odds_api_off_without_key_and_blends_with_fake_events(monkeypatch, tmp_path, caplog):
    import httpx
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    monkeypatch.setattr(httpx, "get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")))
    bs = {"teams": [{"id": 1, "name": "Man Utd", "short_name": "MUN"},
                    {"id": 2, "name": "Coventry City", "short_name": "COV"},
                    {"id": 3, "name": "Spurs", "short_name": "TOT"}]}
    assert odds_api.upcoming_market_lambdas(bs, {}, cache_dir=tmp_path) == {}

    p_h, p_d, p_a, p_o = outcome_probs([2.0], [0.8])
    ev = {"commence_time": "2030-01-01T15:00:00Z", "home_team": "Manchester United", "away_team": "Coventry",
          "bookmakers": [{"key": "x", "markets": [
              {"key": "h2h", "outcomes": [{"name": "Manchester United", "price": 1 / (p_h[0] * 1.05)},
                                          {"name": "Draw", "price": 1 / (p_d[0] * 1.05)},
                                          {"name": "Coventry", "price": 1 / (p_a[0] * 1.05)}]},
              {"key": "totals", "outcomes": [{"name": "Over", "point": 2.5, "price": 1 / (p_o[0] * 1.05)},
                                             {"name": "Under", "point": 2.5, "price": 1 / ((1 - p_o[0]) * 1.05)}]}]}]}
    past = {**ev, "commence_time": "2020-01-01T15:00:00Z", "home_team": "Tottenham Hotspur"}
    unknown = {**ev, "home_team": "Atlantis FC"}
    mk = odds_api.upcoming_market_lambdas(bs, events=[ev, past, unknown])
    assert set(mk) == {(1, 2)}
    assert mk[(1, 2)] == pytest.approx((2.0, 0.8), abs=1e-3)

    # key set but request fails: no blend, and the key never reaches the logs
    monkeypatch.setenv("ODDS_API_KEY", "sekrit-123")
    def boom(*a, **k):
        raise httpx.ConnectError("failed https://api.the-odds-api.com/?apiKey=sekrit-123")
    monkeypatch.setattr(httpx, "get", boom)
    with caplog.at_level(logging.INFO):
        assert odds_api.fetch_events({}, cache_dir=tmp_path) == []
    assert "sekrit" not in caplog.text and "ConnectError" in caplog.text


def test_market_blend_is_geometric():
    res = normalize_results(_two_seasons(5))
    bs = {"teams": [{"id": i + 1, "name": n, "short_name": n} for i, n in enumerate(NEW)]}
    tm = fit_team_model(bs, CFG, results=res)
    base = tm.expected_goals(1, 2, True), tm.expected_goals(2, 1, False)
    tm.market, tm.market_weight = {(1, 2): (3.0, 0.5)}, 0.5
    assert tm.expected_goals(1, 2, True) == pytest.approx(math.sqrt(base[0] * 3.0))
    assert tm.p_clean_sheet(1, 2, True) == pytest.approx(math.exp(-math.sqrt(base[1] * 0.5)))
    assert tm.expected_goals(2, 3, True) == tm.model_expected_goals(2, 3, True)


def test_prior_estimator_recovers_weak_promoted_clubs():
    from fplbot.model.promoted_prior import estimate, season_codes
    assert season_codes(2015, 2017) == ["1516", "1617", "1718"]
    weak = {n: (-0.3 if n.startswith("P") else 0.9 / 17) for n in NEW}
    res = _two_seasons(38, new_att=weak, seed=4)
    est = estimate(res, {"target_weights": {"goals": 0, "xg": 0, "odds": 1}})
    assert est["n_clubs"] == 3 and est["seasons"] == [2026]
    assert est["att"] == pytest.approx(-0.3 - np.mean(list(weak.values())), abs=0.03)
