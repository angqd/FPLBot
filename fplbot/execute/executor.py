"""Turns a plan's first GW into FPL API calls. Dry-run by default.

Payload shapes follow what the FPL web app sends; confirm them once in DevTools (Network tab
-> make a transfer manually -> inspect the POST to /api/transfers/) before setting
execute.dry_run = false, since the API is undocumented and can change between seasons.
"""
from __future__ import annotations

import json


def build_transfer_payload(entry: int, gw: int, plan_gw: dict) -> dict | None:
    ins, outs = plan_gw["transfers_in"], plan_gw["transfers_out"]
    if not ins:
        return None
    # pair same-position in/out so each transfer is a valid like-for-like swap
    outs_left = list(outs)
    pairs = []
    for pin in ins:
        j = next(i for i, po in enumerate(outs_left) if po["pos"] == pin["pos"])
        po = outs_left.pop(j)
        pairs.append({
            "element_in": pin["id"], "element_out": po["id"],
            "purchase_price": int(round(pin["cost"] * 10)),
            "selling_price": int(round(po["sell"] * 10)),
        })
    return {"chip": None, "entry": entry, "event": gw, "transfers": pairs}


def build_lineup_payload(plan_gw: dict) -> dict:
    xi = plan_gw["starting_xi"]
    bench = plan_gw["bench"]  # GK first (FPL requires sub keeper at slot 12)
    cap, vice = plan_gw["captain"]["id"], plan_gw["vice"]["id"]
    picks = [
        {"element": p["id"], "position": i + 1, "is_captain": p["id"] == cap, "is_vice_captain": p["id"] == vice}
        for i, p in enumerate(xi + bench)
    ]
    return {"chip": None, "picks": picks}


def execute(client, entry: int, plan: dict, dry_run: bool = True) -> dict:
    g = plan["gameweeks"][0]
    tx = build_transfer_payload(entry, g["gw"], g)
    lineup = build_lineup_payload(g)
    if dry_run:
        return {"dry_run": True, "transfers": tx, "lineup": lineup}
    results = {}
    if tx:
        results["transfers"] = client.post_transfers(tx)
    results["lineup"] = client.post_lineup(entry, lineup)
    return {"dry_run": False, **json.loads(json.dumps(results, default=str))}
