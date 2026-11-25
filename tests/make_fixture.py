"""Generate a synthetic bootstrap-static + fixtures pair with the real API's field names,
so the pipeline can be tested offline (CI, or when the FPL API is down).

Team strength_attack_*/strength_defence_* are 0, mirroring the live API; a synthetic results
history (football-data.co.uk column names) is generated from the hidden team quality so a
TeamModel can be fitted offline (`make_results()`, written as results.csv)."""
import json
import random
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

random.seed(7)
NEXT_GW = 13


def make():
    teams = []
    for i in range(1, 21):
        q = random.gauss(0, 1)
        teams.append({
            "id": i, "name": f"Club {i}", "short_name": f"C{i:02d}",
            "strength": None,
            "strength_overall_home": 3, "strength_overall_away": 3,
            "strength_attack_home": 0, "strength_attack_away": 0,
            "strength_defence_home": 0, "strength_defence_away": 0,
            "_q": q,
        })
    elements, pid = [], 1
    for t in teams:
        for pos, n in ((1, 3), (2, 8), (3, 9), (4, 5)):
            for k in range(n):
                starter = k < {1: 1, 2: 4, 3: 4, 4: 2}[pos]
                skill = random.gauss(t["_q"] * 0.6, 0.8) + (0.6 if starter else -0.6)
                base = {1: 4.5, 2: 4.5, 3: 5.5, 4: 6.0}[pos]
                cost = max(4.0, round((base + max(0, skill) * {1: 0.6, 2: 1.0, 3: 2.6, 4: 2.8}[pos]) * 2) / 2)
                gp = NEXT_GW - 1
                mins = int(gp * (85 if starter else random.uniform(0, 30)))
                xg90 = max(0, {1: 0, 2: 0.04, 3: 0.18, 4: 0.38}[pos] * (1 + 0.5 * skill))
                xa90 = max(0, {1: 0.01, 2: 0.07, 3: 0.17, 4: 0.12}[pos] * (1 + 0.5 * skill))
                injured = random.random() < 0.05
                elements.append({
                    "id": pid, "web_name": f"P{pid}-{t['short_name']}", "team": t["id"], "element_type": pos,
                    "now_cost": int(cost * 10), "status": "i" if injured else "a",
                    "chance_of_playing_next_round": 0 if injured else None,
                    "news": "Hamstring injury - expected back in 3 weeks" if injured else "",
                    "minutes": mins, "starts": gp if starter else 0,
                    "expected_goals_per_90": round(xg90, 2), "expected_assists_per_90": round(xa90, 2),
                    "defensive_contribution_per_90": round(max(0, random.gauss({1: 0, 2: 9, 3: 7, 4: 3}[pos], 2)), 1),
                    "saves_per_90": round(random.uniform(2, 4), 2) if pos == 1 else 0,
                    "bonus": int(max(0, skill * 4 + 4)) if starter else 0,
                    "ep_next": round(max(0, 2 + skill * 1.5), 1) if starter and not injured else 0,
                    "selected_by_percent": f"{random.uniform(0.1, 30):.1f}",
                    "form": "3.0", "total_points": 0,
                })
                pid += 1
    for t in teams:
        TEAM_QUALITY[t["name"]] = t.pop("_q")
    events = [{"id": g, "is_next": g == NEXT_GW, "finished": g < NEXT_GW,
               "deadline_time": f"2026-12-{min(28, g):02d}T11:00:00Z"} for g in range(1, 39)]
    # double round robin via circle method
    ids = list(range(1, 21))
    rounds = []
    for r in range(19):
        rounds.append([(ids[i], ids[19 - i]) for i in range(10)])
        ids = [ids[0]] + [ids[-1]] + ids[1:-1]
    fixtures, fid = [], 1
    for g, rnd in enumerate(rounds + [[(a, h) for h, a in r] for r in rounds], start=1):
        for h, a in rnd:
            fixtures.append({"id": fid, "event": g, "team_h": h, "team_a": a})
            fid += 1
    return {"elements": elements, "teams": teams, "events": events}, fixtures


TEAM_QUALITY: dict[str, float] = {}  # filled by make(); hidden quality used to simulate results


def _round_robin(names):
    ids = list(names)
    rounds = []
    for _ in range(len(ids) - 1):
        rounds.append([(ids[i], ids[-1 - i]) for i in range(len(ids) // 2)])
        ids = [ids[0]] + [ids[-1]] + ids[1:-1]
    return rounds + [[(a, h) for h, a in r] for r in rounds]


def simulate_results(att: dict, dfn: dict, mu=0.25, home=0.25, seasons=2, current_rounds=0,
                     seed=11, start=date(2024, 8, 17)) -> pd.DataFrame:
    """Poisson results (goals + noisy xG) from known ratings, football-data column layout."""
    rng = np.random.default_rng(seed)
    names = sorted(att)
    rows = []
    rounds = [r for _ in range(seasons) for r in _round_robin(names)] + _round_robin(names)[:current_rounds]
    for k, rnd in enumerate(rounds):
        season, wk = divmod(k, 2 * (len(names) - 1))
        d = start + timedelta(days=365 * season + 7 * wk)
        for h, a in rnd:
            lh = np.exp(mu + home + att[h] - dfn[a])
            la = np.exp(mu + att[a] - dfn[h])
            rows.append({"Date": d.strftime("%d/%m/%Y"), "HomeTeam": h, "AwayTeam": a,
                         "FTHG": int(rng.poisson(lh)), "FTAG": int(rng.poisson(la)),
                         "HxG": round(float(lh * rng.gamma(8, 1 / 8)), 2),
                         "AxG": round(float(la * rng.gamma(8, 1 / 8)), 2)})
    return pd.DataFrame(rows)


def make_results() -> pd.DataFrame:
    """Last season + the current season up to NEXT_GW-1, from the teams make() generated."""
    if not TEAM_QUALITY:
        make()
    att = {n: 0.25 * q for n, q in TEAM_QUALITY.items()}
    dfn = {n: 0.25 * q for n, q in TEAM_QUALITY.items()}
    return simulate_results(att, dfn, seasons=1, current_rounds=NEXT_GW - 1, start=date(2025, 8, 16))


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "fixtures")
    out.mkdir(parents=True, exist_ok=True)
    bs, fx = make()
    (out / "bootstrap.json").write_text(json.dumps(bs))
    (out / "fixtures.json").write_text(json.dumps(fx))
    make_results().to_csv(out / "results.csv", index=False)
    print(f"wrote {len(bs['elements'])} players, {len(fx)} fixtures to {out}")
