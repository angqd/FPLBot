"""Player-level projection: minutes model, shrinkage, DefCon tail, bonus blend, ep_next blend."""
import copy
import math
from datetime import date

import pytest

from fplbot.config import load_config
from fplbot.model.team_strength import fit_team_model
from fplbot.model.xpts import (MinutesProfile, count_tail, expected_floor_div, fixture_points,
                               minutes_profile, model_config, project, rate_priors, return_date,
                               shrunk_rates)
from tests.make_fixture import NEXT_GW, make, make_results

CFG = load_config().raw
MCFG = model_config(CFG["model"])


@pytest.fixture(scope="module")
def world():
    bs, fx = make()
    return bs, fx, fit_team_model(bs, CFG, results=make_results())


def _run(bs, fx, tm, horizon=2, **over):
    cfg = copy.deepcopy(CFG["model"])
    cfg.update(over)
    return project(bs, fx, NEXT_GW, horizon, cfg, team_model=tm).set_index("id")


def _starter(bs, pos):
    return next(p for p in bs["elements"] if p["element_type"] == pos and p["starts"] > 0 and p["status"] == "a")


def test_injured_player_has_zero_xmins(world):
    bs, fx, tm = copy.deepcopy(world[0]), world[1], world[2]
    p = _starter(bs, 4)
    p.update(status="i", chance_of_playing_next_round=0, news="Knee injury - Unknown return date")
    out = _run(bs, fx, tm, blend_fpl_ep=0.0)
    row = out.loc[p["id"]]
    assert row.xmins == 0 and row.p_start == 0 and row[f"xp_{NEXT_GW}"] == 0
    # unknown return date: availability creeps back over later GWs, but stays low
    assert 0 < row[f"xmins_{NEXT_GW + 1}"] < 20
    # left the club: out for the whole horizon
    p.update(status="u", news="Has joined Somewhere FC permanently")
    assert _run(bs, fx, tm).loc[p["id"], f"xmins_{NEXT_GW + 1}"] == 0


def test_doubtful_and_dated_return():
    ref = date(2026, 11, 27)
    assert return_date("Thigh injury - Expected back 20 Dec", ref) == date(2026, 12, 20)
    assert return_date("Suspended until 5 Jan", ref) == date(2027, 1, 5)
    assert return_date("Hamstring injury - Unknown return date", ref) is None
    mc = MCFG["minutes"]
    from fplbot.model.xpts import availability
    p = {"status": "d", "chance_of_playing_next_round": 75, "news": "Knock - 75% chance of playing"}
    assert availability(p, 0, None, ref, mc) == 0.75
    p = {"status": "i", "chance_of_playing_next_round": 0, "news": "Calf injury - Expected back 2 Dec"}
    assert availability(p, 0, date(2026, 11, 27), ref, mc) == 0
    assert availability(p, 1, date(2026, 12, 4), ref, mc) == mc["return_fitness"]


def test_nailed_vs_rotation_and_new_signing():
    mc = MCFG["minutes"]
    nailed = minutes_profile({"minutes": 11 * 88, "starts": 11}, 11, mc)
    rotation = minutes_profile({"minutes": 5 * 70 + 3 * 20, "starts": 5}, 11, mc)
    assert 0.9 < nailed.start_rate < 1 and nailed.p60_start > 0.95
    assert rotation.start_rate < 0.55 and rotation.e_start < nailed.e_start
    # signed after 8 team games: judged on the 3 games since joining, not 11
    from fplbot.model.xpts import _eligible_games
    kick = [f"2026-{m:02d}-15T15:00:00Z" for m in (8, 8, 9, 9, 9, 10, 10, 10, 11, 11, 11)]
    games = _eligible_games({"team_join_date": "2026-11-10"}, kick, 11)
    assert games == 3
    new = minutes_profile({"minutes": 270, "starts": 3}, games, mc)
    assert new.start_rate > 0.8 > minutes_profile({"minutes": 270, "starts": 3}, 11, mc).start_rate


def test_appearance_points_use_p60():
    mc = dict(MCFG["minutes"], _thr={}, _disp=0.0, _bonus_blend=0.0,
              _bonus_event={i: {k: 0.0 for k in ("goal", "assist", "cs", "saves", "defcon", "p60")} for i in (1, 2, 3, 4)})
    mp = MinutesProfile(start_rate=0.8, e_start=70.0, sub_rate=0.5, p60_start=0.75, eligible_games=10)
    zero = {"xg": 0, "xa": 0, "dc": 0, "saves": 0, "bonus": 0}
    scoring = {"goal": {4: 4}, "assist": 3, "cs": {4: 0}, "defcon": {4: 0}}
    fp = fixture_points(4, mp, 1.0, zero, 1.4, 1.4, 0.25, 1.0, 1.0, scoring, mc)
    p_start, p_sub = 0.8, 0.2 * 0.5
    p60 = p_start * 0.75
    assert fp["p60"] == pytest.approx(p60)
    # 2 pts for 60+, 1 pt for any shorter appearance; not a flat 2 for every starter
    assert fp["terms"]["appearance"] == pytest.approx(2 * p60 + (p_start + p_sub - p60))
    assert fp["terms"]["appearance"] == pytest.approx(1.5)
    assert fp["xmins"] == pytest.approx(p_start * 70 + p_sub * mc["sub_mins"])


def test_shrinkage_pulls_low_minutes_toward_prior(world):
    bs = world[0]
    sc = MCFG["shrinkage"]
    priors = rate_priors(bs["elements"], sc)
    base = dict(_starter(bs, 3))
    prior = priors[(3, 3)]  # MID, £7.5-9.5m band
    hot = dict(base, now_cost=80, minutes=90, expected_goals=2.0, expected_goals_per_90=2.0)
    r = shrunk_rates(hot, priors, sc)
    k = sc["k"]["xg"]
    assert r["xg"] == pytest.approx((90 * 2.0 + k * prior["xg"]) / (90 + k))
    assert prior["xg"] < r["xg"] < 0.5  # 2.0 xG/90 in one game is mostly noise
    zero = dict(base, now_cost=80, minutes=0, expected_goals=0, expected_goals_per_90=0, bonus=0)
    assert shrunk_rates(zero, priors, sc)["xg"] == pytest.approx(prior["xg"])
    # price is part of the prior: expensive midfielders get a higher attacking prior
    assert priors[(3, 4)]["xg"] > priors[(3, 0)]["xg"]


def test_defcon_poisson_tail_and_points():
    # P(N >= 10), N ~ Poisson(8) = 1 - sum_{k<10} e^-8 8^k / k!  (hand-computed: 0.283376)
    assert count_tail(8.0, 10) == pytest.approx(0.2833757, abs=1e-6)
    assert count_tail(4.5, 12) == pytest.approx(0.0024043, abs=1e-6)
    assert count_tail(8.0, 10, dispersion=1e6) == pytest.approx(count_tail(8.0, 10), abs=1e-4)
    assert count_tail(8.0, 10, dispersion=5) > count_tail(8.0, 10)  # fatter tail
    # E[floor(X/2)] for Poisson: (lam - P(X odd)) / 2, P(X odd) = (1 - e^-2lam) / 2
    lam = 1.3
    assert expected_floor_div(lam, 2) == pytest.approx((lam - (1 - math.exp(-2 * lam)) / 2) / 2, abs=1e-9)

    mc = dict(MCFG["minutes"], _thr={2: 10}, _disp=0.0, _bonus_blend=0.0,
              _bonus_event={2: {k: 0.0 for k in ("goal", "assist", "cs", "saves", "defcon", "p60")}})
    mp = MinutesProfile(start_rate=1.0, e_start=90.0, sub_rate=0.0, p60_start=1.0, eligible_games=10)
    r = {"xg": 0, "xa": 0, "dc": 8.0, "saves": 0, "bonus": 0}
    scoring = {"goal": {2: 6}, "assist": 3, "cs": {2: 4}, "defcon": {2: 2}}
    fp = fixture_points(2, mp, 1.0, r, 1.4, 1.4, 0.25, 1.0, 1.0, scoring, mc)
    assert fp["terms"]["defcon"] == pytest.approx(2 * 0.2833757, abs=1e-6)  # not 2 * 8/10


def test_bonus_blend_endpoints(world):
    bs, fx, tm = world
    a = _run(bs, fx, tm, blend_fpl_ep=0.0, bonus={"blend": 1.0})
    b = _run(bs, fx, tm, blend_fpl_ep=0.0, bonus={"blend": 0.0})
    pid = _starter(bs, 4)["id"]
    assert a.loc[pid, "model_xp"] != b.loc[pid, "model_xp"]


def test_blend_zero_is_pure_model(world):
    bs, fx, tm = world
    pure = _run(bs, fx, tm, blend_fpl_ep=0.0)
    col = f"xp_{NEXT_GW}"
    assert (pure[col] == pure.model_xp).all()
    full = _run(bs, fx, tm, blend_fpl_ep=1.0)
    ep = {p["id"]: p["ep_next"] for p in bs["elements"]}
    assert all(full.loc[i, col] == pytest.approx(ep[i]) for i in full.index)
    # later GWs never use ep_next
    assert (pure[f"xp_{NEXT_GW + 1}"] == full[f"xp_{NEXT_GW + 1}"]).all()
    # missing ep_next is fine
    bs2 = copy.deepcopy(bs)
    for p in bs2["elements"]:
        p["ep_next"] = None
    assert _run(bs2, fx, tm, blend_fpl_ep=0.35)[col].notna().all()


def test_output_contract(world):
    bs, fx, tm = world
    out = project(bs, fx, NEXT_GW, 3, CFG["model"], team_model=tm)
    xp_cols = [c for c in out.columns if c.startswith("xp_") and c != "xp_total"]
    assert xp_cols == [f"xp_{g}" for g in range(NEXT_GW, NEXT_GW + 3)]
    for c in ("xmins", "p_start", "p60", "model_xp", f"xmins_{NEXT_GW}"):
        assert c in out.columns
    assert out.xmins.between(0, 90).all() and out.p_start.between(0, 1).all()
