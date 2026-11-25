"""Multi-gameweek squad optimiser (MILP, solved with CBC via PuLP).

One model serves both jobs:
  * draft     — empty starting squad, £100m bank, unlimited free moves in GW1 of the plan
  * transfer  — current squad + selling prices + bank + banked FTs, plan the next H GWs

It decides, per GW, the squad, starting XI, captain, transfers, and whether to ROLL. Rolling
isn't a heuristic here: free transfers are a state variable (banked up to 5, 2026/27 rules)
so the solver rolls exactly when "save it and do two next week" scores higher than "do one now".

Free-transfer dynamics (exact, linearised with one binary z_t per GW):
    paid_t = max(0, n_t - ft_t)                    (hits)
    ft_{t+1} = min(cap, ft_t - n_t + paid_t + 1)

Wildcard / Free Hit (SquadState.unlimited_first_gw): the first GW of the plan has unlimited free
transfers (no hits, no max_transfers_per_gw cap), like the draft GW. FT carry-over assumption
(2024/25+ FPL rule): moves made under the chip don't consume banked FTs, so the next GW gets
min(cap, free_transfers + 1) where free_transfers is the count banked *before* the chip. FPL's
my-team hides that count while a chip is active (limit: null), so the CLI passes 1, a
conservative under-estimate. Out of scope: a Free Hit squad reverts to the pre-chip squad the
following GW; the solver models it as a permanent change (i.e. treats it like a Wildcard).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import pulp

from fplbot.config import SQUAD_SHAPE, XI_MAX, XI_MIN


@dataclass
class SquadState:
    picks: dict[int, float] = field(default_factory=dict)  # player_id -> selling price (£m)
    bank: float = 100.0
    free_transfers: int = 1  # banked FTs; with unlimited_first_gw, the count banked before the chip
    unlimited_first_gw: bool = False  # Wildcard / Free Hit active this GW: transfers free + uncapped

    @property
    def is_draft(self) -> bool:
        return not self.picks


def _prune(df: pd.DataFrame, keep: set[int], per_pos: int = 45) -> pd.DataFrame:
    """Keep the strongest candidates per position (by xPts and by xPts/£) + current squad."""
    parts = []
    for pos, g in df.groupby("pos"):
        top = g.nlargest(per_pos, "xp_total")
        val = g.assign(v=g.xp_total / g.cost).nlargest(per_pos // 3, "v")
        parts += [top, val]
    parts.append(df[df.id.isin(keep)])
    return pd.concat(parts).drop_duplicates("id").reset_index(drop=True)


def solve(
    proj: pd.DataFrame,
    state: SquadState,
    cfg: dict,
    exclude_first_moves: list[frozenset] | None = None,
    time_limit: int = 120,
) -> dict:
    gws = sorted(int(c[3:]) for c in proj.columns if c.startswith("xp_") and c != "xp_total")
    t0 = gws[0]
    df = _prune(proj, set(state.picks))
    P = list(df.id)
    row = df.set_index("id")
    xp = {(p, t): float(row.at[p, f"xp_{t}"]) for p in P for t in gws}
    cost = {p: float(row.at[p, "cost"]) for p in P}
    sell = {p: state.picks.get(p, cost[p]) for p in P}
    pos = {p: int(row.at[p, "pos"]) for p in P}
    team = {p: int(row.at[p, "team"]) for p in P}

    m = pulp.LpProblem("fpl", pulp.LpMaximize)
    B = pulp.LpBinary
    sq = pulp.LpVariable.dicts("sq", (P, gws), cat=B)
    xi = pulp.LpVariable.dicts("xi", (P, gws), cat=B)
    cap = pulp.LpVariable.dicts("cap", (P, gws), cat=B)
    buy = pulp.LpVariable.dicts("buy", (P, gws), cat=B)
    out = pulp.LpVariable.dicts("sell", (P, gws), cat=B)
    bank = pulp.LpVariable.dicts("bank", gws + [t0 - 1], lowBound=0)
    n = pulp.LpVariable.dicts("n", gws, lowBound=0, cat="Integer")
    ft = pulp.LpVariable.dicts("ft", gws + [gws[-1] + 1], lowBound=0, upBound=cfg["max_free_transfers"], cat="Integer")
    paid = pulp.LpVariable.dicts("paid", gws, lowBound=0, cat="Integer")
    z = pulp.LpVariable.dicts("z", gws, cat=B)
    M = 15

    m += bank[t0 - 1] == state.bank
    m += ft[t0] == (0 if state.is_draft else state.free_transfers)

    for i, t in enumerate(gws):
        prev = gws[i - 1] if i else None
        for p in P:
            before = sq[p][prev] if prev else (1 if p in state.picks else 0)
            m += sq[p][t] == before + buy[p][t] - out[p][t]
            m += buy[p][t] + out[p][t] <= 1
            m += xi[p][t] <= sq[p][t]
            m += cap[p][t] <= xi[p][t]
        for k, cnt in SQUAD_SHAPE.items():
            m += pulp.lpSum(sq[p][t] for p in P if pos[p] == k) == cnt
            m += pulp.lpSum(xi[p][t] for p in P if pos[p] == k) >= XI_MIN[k]
            m += pulp.lpSum(xi[p][t] for p in P if pos[p] == k) <= XI_MAX[k]
        m += pulp.lpSum(xi[p][t] for p in P) == 11
        m += pulp.lpSum(cap[p][t] for p in P) == 1
        for c in set(team.values()):
            m += pulp.lpSum(sq[p][t] for p in P if team[p] == c) <= cfg["max_per_club"]
        bprev = bank[prev] if prev else bank[t0 - 1]
        m += bank[t] == bprev + pulp.lpSum(sell[p] * out[p][t] - cost[p] * buy[p][t] for p in P)

        m += n[t] == pulp.lpSum(buy[p][t] for p in P)
        if state.is_draft and i == 0:  # initial squad build: free, unlimited
            m += paid[t] == 0
            m += ft[gws[1] if len(gws) > 1 else t + 1] == 1
            m += z[t] == 0
            continue
        if state.unlimited_first_gw and i == 0:  # wildcard / free hit: free, uncapped moves
            m += paid[t] == 0
            m += z[t] == 0
            # banked FTs carry through the chip (+1). Free Hit revert next GW: not modelled.
            m += ft[gws[1] if len(gws) > 1 else t + 1] <= state.free_transfers + 1
            continue
        m += n[t] <= cfg["max_transfers_per_gw"]
        m += paid[t] >= n[t] - ft[t]
        m += paid[t] <= n[t] - ft[t] + M * (1 - z[t])
        m += paid[t] <= M * z[t]
        if not cfg["allow_hits"]:
            m += paid[t] == 0
        nxt = gws[i + 1] if i + 1 < len(gws) else gws[-1] + 1
        m += ft[nxt] <= ft[t] - n[t] + paid[t] + 1

    if exclude_first_moves:  # no-good cuts → distinct candidate plans for the judge
        for moves in exclude_first_moves:
            if moves:
                m += pulp.lpSum(buy[p][t0] for p in moves if p in buy) <= len(moves) - 1
            else:
                m += n[t0] >= 1

    obj = []
    for i, t in enumerate(gws):
        w = cfg["decay"] ** i
        obj.append(w * pulp.lpSum(xp[p, t] * (xi[p][t] + cap[p][t] + cfg["bench_weight"] * (sq[p][t] - xi[p][t])) for p in P))
        obj.append(-w * cfg["hit_cost"] * paid[t])
        if not (state.is_draft and i == 0):
            obj.append(-cfg["min_gain_per_transfer"] * n[t])
    obj.append(cfg["ft_value"] * ft[gws[-1] + 1])
    m += pulp.lpSum(obj)

    status = m.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit))
    if pulp.LpStatus[status] not in ("Optimal", "Not Solved") or m.objective.value() is None:
        raise RuntimeError(f"Solver failed: {pulp.LpStatus[status]}")

    v = lambda x: round(x.value() or 0)  # noqa: E731
    plan = {"objective": round(m.objective.value(), 2), "gameweeks": []}
    for t in gws:
        squad = [p for p in P if v(sq[p][t])]
        starters = [p for p in squad if v(xi[p][t])]
        bench = sorted((p for p in squad if p not in starters), key=lambda p: (pos[p] != 1, -xp[p, t]))
        capt = next(p for p in starters if v(cap[p][t]))
        vice = max((p for p in starters if p != capt), key=lambda p: xp[p, t])
        info = lambda p: {"id": p, "name": row.at[p, "name"], "pos": pos[p], "cost": cost[p], "xp": round(xp[p, t], 2)}  # noqa: E731
        plan["gameweeks"].append({
            "gw": t,
            "free_transfers": v(ft[t]),
            "transfers_in": [info(p) for p in P if v(buy[p][t])],
            "transfers_out": [{**info(p), "sell": sell[p]} for p in P if v(out[p][t])],
            "hits": v(paid[t]),
            "rolled": v(n[t]) == 0 and not ((state.is_draft or state.unlimited_first_gw) and t == t0),
            "starting_xi": [info(p) for p in sorted(starters, key=lambda p: pos[p])],
            "bench": [info(p) for p in bench],
            "captain": info(capt), "vice": info(vice),
            "bank": round(bank[t].value(), 1),
            "xp_xi": round(sum(xp[p, t] for p in starters) + xp[capt, t], 2),
        })
    return plan


def candidate_plans(proj: pd.DataFrame, state: SquadState, cfg: dict, k: int) -> list[dict]:
    """Best plan plus k-1 runner-ups that differ in *this week's* move set."""
    plans, seen = [], []
    for _ in range(k):
        try:
            plan = solve(proj, state, cfg, exclude_first_moves=seen)
        except RuntimeError:
            break
        plans.append(plan)
        seen.append(frozenset(x["id"] for x in plan["gameweeks"][0]["transfers_in"]))
    return plans
