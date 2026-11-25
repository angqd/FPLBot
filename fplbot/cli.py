"""fplbot CLI.

  fplbot ingest            pull bootstrap + fixtures, snapshot to data/
  fplbot draft             build the best £100m squad from scratch for the next GW
  fplbot plan              plan transfers/roll for your real team (needs auth)
  fplbot run               the weekly job: ingest -> plan -> (judge) -> execute if in window
--offline PATH uses a saved bootstrap/fixtures pair instead of the API: a dir with
{bootstrap,fixtures}.json (tests/fixtures), an ingest snapshot dir (data/raw/<ts>/ with .json.gz),
data/raw itself (newest snapshot), or the literal `latest` (newest snapshot in DATA_DIR/raw).
--profile NAME (before the command) loads ROOT/.env.NAME and keeps decisions in DATA_DIR/NAME/.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import typer

from fplbot import config as fpl_config
from fplbot.config import POSITIONS, load_config
from fplbot.data import store
from fplbot.data.fpl_client import FPLClient
from fplbot.model.xpts import project
from fplbot.optimize.solver import SquadState, candidate_plans, solve

log = logging.getLogger(__name__)
app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.callback()
def main(profile: str = typer.Option(None, "--profile", envvar="FPLBOT_PROFILE",
                                     help="Use ROOT/.env.NAME and a separate decisions DB (second account).")):
    if profile:
        try:
            env_file = fpl_config.activate_profile(profile)
        except ValueError as e:
            raise typer.BadParameter(str(e), param_hint="--profile") from e
        if not env_file.exists():
            raise typer.BadParameter(f"{env_file} does not exist", param_hint="--profile")
        typer.echo(f"[profile {profile}] env={env_file.name} state={fpl_config.STATE_DIR}", err=True)


def _load(offline: Path | None, save: bool = True):
    if offline:
        if str(offline) == "latest":
            return store.load_latest()
        return store.load_dir(offline)
    c = FPLClient()
    bs, fx = c.bootstrap(), c.fixtures()
    if save:
        store.save_snapshot(bs, fx)
    return bs, fx


def _print_plan(plan: dict, first_only: bool = False):
    for g in plan["gameweeks"][: 1 if first_only else None]:
        typer.secho(f"\n=== GW{g['gw']}  FT={g['free_transfers']}  bank=£{g['bank']}m  XI xPts={g['xp_xi']}", bold=True)
        if g["rolled"]:
            typer.echo("  ROLL transfer")
        for po, pi in zip(g["transfers_out"], g["transfers_in"]):
            typer.echo(f"  OUT {po['name']:<16} £{po['sell']:<5} ->  IN {pi['name']:<16} £{pi['cost']:<5} xp {pi['xp']}")
        if g["hits"]:
            typer.secho(f"  hits: -{4 * g['hits']}", fg="red")
        if first_only or g is plan["gameweeks"][0]:
            for p in g["starting_xi"]:
                tag = " (C)" if p["id"] == g["captain"]["id"] else " (V)" if p["id"] == g["vice"]["id"] else ""
                typer.echo(f"   {POSITIONS[p['pos']]:<3} {p['name']:<18} £{p['cost']:<5} xp {p['xp']}{tag}")
            typer.echo("   bench: " + ", ".join(p["name"] for p in g["bench"]))


def parse_my_team(t: dict) -> tuple[dict[int, float], float, int | None, bool]:
    """Parse a my-team/{entry}/ payload -> (picks {element: selling £m}, bank £m, ft, unlimited).

    Free transfers follow the FPL web app: `limit is None ? 0 : max(limit - made, 0)`.
    `limit: null` or `status: "unlimited"` means a wildcard / free hit is active, so transfers
    are not capped: returns ft=None, unlimited=True. Selling prices come only from
    picks[].selling_price (never now_cost: that would break the budget constraint)."""
    picks = {}
    for p in t["picks"]:
        sp = p.get("selling_price")
        if sp is None:
            raise ValueError(f"my-team pick {p.get('element')} has no selling_price; refusing to guess "
                             "(now_cost would overstate the budget)")
        picks[p["element"]] = sp / 10
    tr = t.get("transfers") or {}
    if "bank" not in tr:
        raise ValueError("my-team payload has no transfers.bank")
    limit = tr.get("limit")
    unlimited = limit is None or tr.get("status") == "unlimited"
    ft = None if unlimited else max(limit - tr.get("made", 0), 0)
    return picks, tr["bank"] / 10, ft, unlimited


def _state_from_api(client: FPLClient, entry: int, max_free_transfers: int = 5) -> SquadState:
    picks, bank, ft, unlimited = parse_my_team(client.my_team(entry))
    if unlimited:
        # SquadState has no "unlimited" flag yet; the cap is the closest approximation and
        # under-uses the chip (it still treats moves beyond the cap as hits / disallowed).
        msg = (f"Transfers are UNCAPPED this GW (wildcard/free hit active). Planning with "
               f"free_transfers={max_free_transfers} as an approximation, not truly unlimited.")
        log.warning(msg)
        typer.secho("WARNING: " + msg, fg="yellow", err=True)
        ft = max_free_transfers
    return SquadState(picks=picks, bank=bank, free_transfers=ft)


@app.command()
def ingest(offline: Path = typer.Option(None)):
    bs, _ = _load(offline)
    typer.echo(f"Snapshot saved. Next GW: {store.current_gw(bs)}  players: {len(bs['elements'])}")


@app.command()
def draft(offline: Path = typer.Option(None), config: Path = typer.Option(None), out: Path = typer.Option(None)):
    cfg = load_config(config).raw
    bs, fx = _load(offline)
    gw = store.current_gw(bs)
    proj = project(bs, fx, gw, cfg["horizon"], cfg["model"])
    plan = solve(proj, SquadState(bank=cfg["budget"]), cfg)
    _print_plan(plan, first_only=True)
    if out:
        out.write_text(json.dumps(plan, indent=1))


@app.command()
def plan(offline: Path = typer.Option(None), config: Path = typer.Option(None),
         state_file: Path = typer.Option(None, help="JSON SquadState instead of live my-team")):
    cfg = load_config(config)
    bs, fx = _load(offline)
    gw = store.current_gw(bs)
    if state_file:
        s = json.loads(state_file.read_text())
        state = SquadState({int(k): v for k, v in s["picks"].items()}, s["bank"], s["free_transfers"])
    else:
        from fplbot.data.auth import OIDCAuth
        state = _state_from_api(FPLClient(auth=OIDCAuth()), cfg.entry_id, cfg.max_free_transfers)
    proj = project(bs, fx, gw, cfg.horizon, cfg.model)
    _print_plan(solve(proj, state, cfg.raw))


@app.command()
def run(config: Path = typer.Option(None), force: bool = typer.Option(False, help="ignore deadline window")):
    """Weekly job. Safe to schedule hourly/daily: it only acts inside the pre-deadline window,
    once per GW, and only submits when execute.dry_run is false."""
    from fplbot.data.auth import OIDCAuth
    from fplbot.execute.executor import execute
    from fplbot.judge.llm_judge import judge

    cfg = load_config(config)
    client = FPLClient(auth=OIDCAuth())
    bs, fx = _load(None)
    gw = store.current_gw(bs)
    deadline = datetime.fromisoformat(next(e for e in bs["events"] if e["id"] == gw)["deadline_time"].replace("Z", "+00:00"))
    now = datetime.now(timezone.utc)
    window = timedelta(hours=cfg.execute["hours_before_deadline"])
    if not force and not (deadline - window <= now < deadline):
        typer.echo(f"GW{gw} deadline {deadline:%a %d %b %H:%M} UTC — outside decision window; snapshot only.")
        raise typer.Exit()
    if store.already_executed(gw):
        typer.echo(f"GW{gw} already executed.")
        raise typer.Exit()

    state = _state_from_api(client, cfg.entry_id, cfg.max_free_transfers)
    proj = project(bs, fx, gw, cfg.horizon, cfg.model)
    j = cfg.judge
    plans = candidate_plans(proj, state, cfg.raw, j["candidates"] if j["enabled"] else 1)
    idx, reason = (judge(plans, dict(zip(proj.id, proj.news)), j["model"]) if j["enabled"] else (0, "solver optimum"))
    chosen = plans[idx]
    _print_plan(chosen, first_only=True)
    typer.echo(f"\nChosen plan #{idx}: {reason}")
    result = execute(client, cfg.entry_id, chosen, dry_run=cfg.execute["dry_run"])
    store.record_decision(gw, {"plan": chosen, "judge": reason, "result": result}, executed=not result["dry_run"])
    typer.echo(json.dumps(result, indent=1)[:2000])


if __name__ == "__main__":
    app()
