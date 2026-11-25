"""Generate a synthetic bootstrap-static + fixtures pair with the real API's field names,
so the pipeline can be tested offline (CI, or when the FPL API is down)."""
import json
import random
import sys
from pathlib import Path

random.seed(7)
NEXT_GW = 13


def make():
    teams = []
    for i in range(1, 21):
        q = random.gauss(0, 1)
        teams.append({
            "id": i, "name": f"Club {i}", "short_name": f"C{i:02d}",
            "strength_attack_home": int(1150 + 90 * q + random.gauss(0, 20)),
            "strength_attack_away": int(1110 + 90 * q + random.gauss(0, 20)),
            "strength_defence_home": int(1150 + 90 * q + random.gauss(0, 20)),
            "strength_defence_away": int(1110 + 90 * q + random.gauss(0, 20)),
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
        del t["_q"]
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


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "fixtures")
    out.mkdir(parents=True, exist_ok=True)
    bs, fx = make()
    (out / "bootstrap.json").write_text(json.dumps(bs))
    (out / "fixtures.json").write_text(json.dumps(fx))
    print(f"wrote {len(bs['elements'])} players, {len(fx)} fixtures to {out}")
