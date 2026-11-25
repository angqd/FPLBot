"""Optional LLM-as-judge layer.

Design rule: the LLM never invents moves. The solver produces K valid, budget-checked plans;
the LLM only *chooses among them*, using soft context the maths can't see (press-conference
news, rotation risk, "flagged but trained today"). If the LLM call fails or returns garbage,
we fall back to plan 0 (the solver's optimum) — so the bot never depends on the LLM to act.
"""
from __future__ import annotations

import json
import os
import re

SYSTEM = (
    "You are the final reviewer for an automated Fantasy Premier League manager. You will get "
    "K candidate plans produced by an optimiser, each already legal and within budget, plus "
    "player news. Pick the single best plan for THIS gameweek's move. Weigh injury/rotation "
    "news and fixture context the optimiser may underrate. You may only choose an index; you "
    "cannot modify a plan. Reply with JSON only: {\"choice\": <int>, \"reason\": \"<2 sentences>\"}"
)


def _summarise(plan: dict, news: dict[int, str]) -> dict:
    gw = plan["gameweeks"][0]
    ids = [p["id"] for p in gw["transfers_in"] + gw["starting_xi"]]
    return {
        "objective_xpts": plan["objective"],
        "this_gw": {
            "in": [f"{p['name']} £{p['cost']}m xp={p['xp']}" for p in gw["transfers_in"]],
            "out": [f"{p['name']}" for p in gw["transfers_out"]],
            "rolled": gw["rolled"], "hits": gw["hits"],
            "captain": gw["captain"]["name"],
        },
        "later_moves": [
            {"gw": g["gw"], "in": [p["name"] for p in g["transfers_in"]], "out": [p["name"] for p in g["transfers_out"]]}
            for g in plan["gameweeks"][1:]
        ],
        "news": {pid: news[pid] for pid in ids if news.get(pid)},
    }


def judge(plans: list[dict], news: dict[int, str], model: str) -> tuple[int, str]:
    if len(plans) <= 1:
        return 0, "single candidate"
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        payload = [{"index": i, **_summarise(p, news)} for i, p in enumerate(plans)]
        msg = client.messages.create(
            model=model,
            max_tokens=400,
            system=SYSTEM,
            messages=[{"role": "user", "content": json.dumps(payload, indent=1)}],
        )
        text = msg.content[0].text
        data = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
        choice = int(data["choice"])
        if not 0 <= choice < len(plans):
            raise ValueError(choice)
        return choice, data.get("reason", "")
    except Exception as e:  # never block execution on the judge
        return 0, f"judge unavailable ({type(e).__name__}); using solver optimum"
