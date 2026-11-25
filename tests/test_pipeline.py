from collections import Counter

import pytest

from fplbot.config import SQUAD_SHAPE, load_config
from fplbot.execute.executor import build_lineup_payload, build_transfer_payload
from fplbot.model.xpts import project
from fplbot.optimize.solver import SquadState, candidate_plans, solve
from tests.make_fixture import NEXT_GW, make

CFG = load_config().raw


@pytest.fixture(scope="module")
def data():
    bs, fx = make()
    proj = project(bs, fx, NEXT_GW, 3, CFG["model"])
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
