from collections import Counter

import pytest

from fplbot.config import SQUAD_SHAPE, load_config
from fplbot.execute.executor import build_lineup_payload, build_transfer_payload
from fplbot.model.team_strength import fit_team_model
from fplbot.model.xpts import project
from fplbot.optimize.solver import SquadState, candidate_plans, solve
from tests.make_fixture import NEXT_GW, make, make_results

CFG = load_config().raw


@pytest.fixture(scope="module")
def data():
    bs, fx = make()
    tm = fit_team_model(bs, CFG, results=make_results())
    proj = project(bs, fx, NEXT_GW, 3, CFG["model"], team_model=tm)
    return bs, fx, proj


def _check_squad(gw, team_of):
    squad = gw["starting_xi"] + gw["bench"]
    assert len(squad) == 15 and len(gw["starting_xi"]) == 11
    assert Counter(p["pos"] for p in squad) == Counter(SQUAD_SHAPE)
    assert max(Counter(team_of[p["id"]] for p in squad).values()) <= 3
    assert gw["bench"][0]["pos"] == 1  # sub keeper in slot 12


def test_injured_players_project_low(data):
    _, _, proj = data
    assert proj[proj.status == "i"].xp_total.max() < 1.0


def test_draft_is_legal_and_within_budget(data):
    bs, _, proj = data
    team_of = {p["id"]: p["team"] for p in bs["elements"]}
    plan = solve(proj, SquadState(bank=100.0), CFG)
    g = plan["gameweeks"][0]
    _check_squad(g, team_of)
    assert sum(p["cost"] for p in g["starting_xi"] + g["bench"]) <= 100.0 + 1e-6
    assert g["bank"] >= 0


def test_transfer_mode_respects_ft_and_rolls(data):
    bs, _, proj = data
    team_of = {p["id"]: p["team"] for p in bs["elements"]}
    draft = solve(proj, SquadState(bank=100.0), CFG)["gameweeks"][0]
    picks = {p["id"]: p["cost"] for p in draft["starting_xi"] + draft["bench"]}
    state = SquadState(picks=picks, bank=draft["bank"], free_transfers=1)
    plan = solve(proj, state, CFG)
    for g in plan["gameweeks"]:
        _check_squad(g, team_of)
        assert len(g["transfers_in"]) <= CFG["max_transfers_per_gw"]
        assert g["hits"] == 0  # allow_hits: false
        assert len(g["transfers_in"]) <= g["free_transfers"]


def test_candidates_are_distinct_and_payloads_build(data):
    _, _, proj = data
    draft = solve(proj, SquadState(bank=100.0), CFG)["gameweeks"][0]
    picks = {p["id"]: p["cost"] for p in draft["starting_xi"] + draft["bench"]}
    state = SquadState(picks=picks, bank=draft["bank"], free_transfers=2)
    plans = candidate_plans(proj, state, CFG, 3)
    firsts = [frozenset(p["id"] for p in pl["gameweeks"][0]["transfers_in"]) for pl in plans]
    assert len(set(firsts)) == len(firsts)
    for pl in plans:
        g = pl["gameweeks"][0]
        tx = build_transfer_payload(123, g["gw"], g)
        if tx:
            assert len(tx["transfers"]) == len(g["transfers_in"])
        lu = build_lineup_payload(g)
        assert len(lu["picks"]) == 15 and sum(p["is_captain"] for p in lu["picks"]) == 1


def _weak_squad(bs, proj):
    """Legal 15 of the lowest-xPts players (<=3 per club), so a wildcard wants many moves."""
    team_of = {p["id"]: p["team"] for p in bs["elements"]}
    picks, clubs = {}, Counter()
    for pos, cnt in SQUAD_SHAPE.items():
        got = 0
        for r in proj[proj.pos == pos].sort_values("xp_total").itertuples():
            if got == cnt:
                break
            if clubs[team_of[r.id]] < 3:
                picks[r.id] = float(r.cost)
                clubs[team_of[r.id]] += 1
                got += 1
    return picks, team_of


def test_unlimited_first_gw_allows_many_free_transfers(data):
    bs, _, proj = data
    picks, team_of = _weak_squad(bs, proj)
    bank = 100.0 - sum(picks.values())
    wc = solve(proj, SquadState(picks=picks, bank=bank, free_transfers=1, unlimited_first_gw=True), CFG)
    g0 = wc["gameweeks"][0]
    _check_squad(g0, team_of)
    assert len(g0["transfers_in"]) > 5 > CFG["max_transfers_per_gw"]
    assert g0["hits"] == 0 and not g0["rolled"]
    for g in wc["gameweeks"][1:]:  # normal rules again after the chip
        _check_squad(g, team_of)
        assert g["hits"] == 0 and len(g["transfers_in"]) <= CFG["max_transfers_per_gw"]
        assert len(g["transfers_in"]) <= g["free_transfers"] <= CFG["max_free_transfers"]
    assert wc["gameweeks"][1]["free_transfers"] <= 2  # banked 1 carries through the chip, +1

    normal = solve(proj, SquadState(picks=picks, bank=bank, free_transfers=1), CFG)
    n0 = normal["gameweeks"][0]
    assert len(n0["transfers_in"]) <= 1 and n0["hits"] == 0
    assert wc["objective"] > normal["objective"]



def test_bench_boost_and_forced_transfers(data):
    """transfers_by_gw forces an exact move count (hits allowed); bench_boost_gw values the bench fully."""
    _, _, proj = data
    t0, t1 = NEXT_GW, NEXT_GW + 1
    d = solve(proj, SquadState(bank=100.0), CFG)["gameweeks"][0]
    picks = {p["id"]: p["cost"] for p in d["starting_xi"] + d["bench"]}
    st = SquadState(picks=picks, bank=d["bank"], free_transfers=1)
    forced = solve(proj, st, {**CFG, "allow_hits": True, "transfers_by_gw": {t0: 3}})["gameweeks"][0]
    assert len(forced["transfers_in"]) == 3 and forced["hits"] == 2
    bench = lambda c: sum(p["xp"] for p in solve(proj, SquadState(bank=100.0), c)["gameweeks"][1]["bench"])  # noqa: E731
    assert bench({**CFG, "bench_boost_gw": t1}) >= bench(CFG) - 1e-6
