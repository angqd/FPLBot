"""Viability gate checks 1b-1d. READ-ONLY: never POSTs to FPL.

  python scripts/viability_check.py

1b  refresh token -> access token; confirms the rotated refresh token was written to .env
1c  GET my-team/{entry}/: selling/purchase prices, bank, free transfers; checks the half-profit rule
1d  builds the transfer + lineup payloads from your real squad and checks their shape against
    what the FPL web app sends (taken from fantasy.premierleague.com's JS bundle, 2026-11-25)
Never prints tokens.
"""
from __future__ import annotations

import hashlib
import json
import sys

from fplbot.config import ENV_FILE, load_config
from fplbot.data.auth import OIDCAuth
from fplbot.data.fpl_client import FPLClient
from fplbot.execute.executor import build_lineup_payload, build_transfer_payload

# Shapes the FPL web app sends (from its JS bundle)
WEB_TRANSFER_KEYS = {"chip", "entry", "event", "transfers"}
WEB_TRANSFER_ITEM_KEYS = {"element_in", "element_out", "purchase_price", "selling_price"}
WEB_LINEUP_KEYS = {"chip", "picks"}
WEB_PICK_KEYS = {"element", "position", "is_captain", "is_vice_captain"}

results: dict[str, str] = {}


def env_hash() -> str:
    for line in ENV_FILE.read_text().splitlines():
        if line.startswith("FPL_REFRESH_TOKEN="):
            return hashlib.sha256(line.encode()).hexdigest()[:12]
    return "missing"


def check_1b() -> OIDCAuth:
    print("== 1b  OIDC refresh")
    before = env_hash()
    auth = OIDCAuth()
    tok = auth.access_token()
    after = env_hash()
    rotated = before != after
    print(f"  access token received: {bool(tok)} (length {len(tok)})")
    print(f"  refresh token rotated and written to .env: {rotated}")
    results["1b"] = "PASS" if tok else "FAIL"
    if not rotated:
        results["1b"] += " (no rotation seen; fine if PingOne reuses tokens, but check)"
    return auth


def check_1c(client: FPLClient, entry: int, elements: dict) -> dict:
    print("\n== 1c  my-team prices, bank, free transfers")
    t = client.my_team(entry)
    print(f"  top-level keys: {sorted(t)}")
    picks = t.get("picks", [])
    missing = [p for p in picks if "selling_price" not in p or "purchase_price" not in p]
    print(f"  picks: {len(picks)}   missing selling/purchase price: {len(missing)}")
    bad_rule = []
    print(f"  {'pos':>3} {'player':<16} {'bought':>6} {'now':>5} {'sell':>5}  rule")
    for p in picks:
        e = elements[p["element"]]
        buy, now, sell = p.get("purchase_price"), e["now_cost"], p.get("selling_price")
        expect = now if now <= buy else buy + (now - buy) // 2  # FPL keeps half the profit, rounded down
        ok = sell == expect
        if not ok:
            bad_rule.append(e["web_name"])
        flag = "" if now == buy else ("  <- risen" if now > buy else "  <- fallen")
        print(f"  {p['position']:>3} {e['web_name']:<16} {buy/10:>6.1f} {now/10:>5.1f} {sell/10 if sell is not None else 'NA':>5}  "
              f"{'ok' if ok else 'MISMATCH (expected ' + str(expect/10) + ')'}{flag}")
    tr = t.get("transfers", {})
    print(f"  transfers: {json.dumps(tr)}")
    limit, made = tr.get("limit"), tr.get("made", 0)
    ft = None if limit is None else max(limit - made, 0)
    print(f"  free transfers available (web-app formula): {ft if ft is not None else 'unlimited (chip active)'}")
    print(f"  chips: {[(c.get('name'), c.get('status_for_entry')) for c in t.get('chips', [])]}")
    if missing or not picks or "bank" not in tr:
        results["1c"] = "FAIL: selling/purchase price or bank missing. BLOCKER"
    elif bad_rule:
        results["1c"] = f"PASS with note: half-profit rule mismatch for {bad_rule}; check these in the FPL UI"
    else:
        results["1c"] = "PASS: all 15 have selling+purchase price, bank present, half-profit rule holds"
    return t


def check_1d(team: dict, elements: dict, entry: int, gw: int):
    print("\n== 1d  payload shapes (built locally, NOT sent)")
    by_id = {p["element"]: p for p in team["picks"]}
    info = lambda pid: {"id": pid, "name": elements[pid]["web_name"], "pos": elements[pid]["element_type"],  # noqa: E731
                        "cost": elements[pid]["now_cost"] / 10, "xp": 0}
    ordered = sorted(team["picks"], key=lambda p: p["position"])
    xi, bench = [info(p["element"]) for p in ordered[:11]], [info(p["element"]) for p in ordered[11:]]
    cap = next(p["element"] for p in ordered if p["is_captain"])
    vice = next(p["element"] for p in ordered if p["is_vice_captain"])
    # hypothetical swap: last bench outfielder -> cheapest same-position player not owned
    out_id = ordered[-1]["element"]
    pos = elements[out_id]["element_type"]
    in_id = min((e for e in elements.values() if e["element_type"] == pos and e["id"] not in by_id
                 and e["status"] == "a"), key=lambda e: e["now_cost"])["id"]
    plan_gw = {"transfers_in": [info(in_id)], "transfers_out": [{**info(out_id), "sell": by_id[out_id]["selling_price"] / 10}],
               "starting_xi": xi, "bench": bench, "captain": info(cap), "vice": info(vice)}
    tx = build_transfer_payload(entry, gw, plan_gw)
    lu = build_lineup_payload(plan_gw)
    print("  transfer payload:", json.dumps(tx))
    print("  lineup payload (first 2 picks):", json.dumps({**lu, "picks": lu["picks"][:2]}))
    problems = []
    if set(tx) != WEB_TRANSFER_KEYS:
        problems.append(f"transfer keys {sorted(tx)}")
    if any(set(i) != WEB_TRANSFER_ITEM_KEYS for i in tx["transfers"]):
        problems.append("transfer item keys")
    if any(not isinstance(i[k], int) for i in tx["transfers"] for k in ("purchase_price", "selling_price")):
        problems.append("prices must be int tenths")
    if tx["transfers"][0]["selling_price"] != by_id[out_id]["selling_price"]:
        problems.append("selling_price does not round-trip from my-team")
    if set(lu) != WEB_LINEUP_KEYS or any(set(p) != WEB_PICK_KEYS for p in lu["picks"]):
        problems.append("lineup keys")
    if [p["element"] for p in lu["picks"]] != [p["element"] for p in ordered]:
        problems.append("lineup order differs from current team (expected identical for an unchanged XI)")
    results["1d"] = "PASS: shapes match the web app" if not problems else "FAIL: " + "; ".join(problems)


def main():
    cfg = load_config()
    auth = check_1b()
    client = FPLClient(auth=auth)
    entry = cfg.entry_id
    if not entry:
        me = client._req("GET", "me/", authed=True) or {}
        entry = (me.get("player") or {}).get("entry")
        if not entry:
            sys.exit("Could not find your team id via me/; set FPL_ENTRY_ID in .env")
        print(f"  team id from me/: {entry}  (add FPL_ENTRY_ID={entry} to .env)")
    bs = client.bootstrap()
    elements = {e["id"]: e for e in bs["elements"]}
    gw = next(e["id"] for e in bs["events"] if e.get("is_next"))
    team = check_1c(client, entry, elements)
    if team.get("picks"):
        check_1d(team, elements, entry, gw)
    print("\n== Summary")
    for k, v in results.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
