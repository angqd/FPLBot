"""Snapshot store. Every ingest writes the raw JSON (gzipped, for replay/backtests) and a
flattened row per player into SQLite so price/form history accumulates week over week."""
from __future__ import annotations

import gzip
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from fplbot import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS player_snapshots (
  snapshot_at TEXT, gw INTEGER, player_id INTEGER, web_name TEXT, team INTEGER,
  position INTEGER, now_cost REAL, selected_by REAL, status TEXT, chance_next REAL,
  form REAL, total_points INTEGER, minutes INTEGER, xg90 REAL, xa90 REAL, ep_next REAL,
  transfers_in_event INTEGER, transfers_out_event INTEGER,
  PRIMARY KEY (snapshot_at, player_id)
);
CREATE TABLE IF NOT EXISTS decisions (
  gw INTEGER PRIMARY KEY, decided_at TEXT, plan_json TEXT, executed INTEGER DEFAULT 0
);
"""


def _db(path: Path | None = None) -> sqlite3.Connection:
    """Shared DB (public player snapshots) by default; pass _state_db_path() for decisions."""
    path = path or config.DATA_DIR / "fplbot.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    return con


def _state_db_path() -> Path:
    """Per-profile DB for decisions, so two accounts' idempotency records never collide.
    Default profile: DATA_DIR/fplbot.sqlite (unchanged); profile X: DATA_DIR/X/fplbot.sqlite."""
    return config.STATE_DIR / "fplbot.sqlite"


def current_gw(bootstrap: dict) -> int:
    """The next GW to plan for (first event not yet finished/started past deadline)."""
    for ev in bootstrap["events"]:
        if ev.get("is_next"):
            return ev["id"]
    for ev in bootstrap["events"]:
        if not ev.get("finished"):
            return ev["id"]
    return bootstrap["events"][-1]["id"]


def save_snapshot(bootstrap: dict, fixtures: list[dict]) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    raw = config.DATA_DIR / "raw" / ts
    raw.mkdir(parents=True, exist_ok=True)
    for name, obj in (("bootstrap", bootstrap), ("fixtures", fixtures)):
        with gzip.open(raw / f"{name}.json.gz", "wt") as f:
            json.dump(obj, f)

    gw = current_gw(bootstrap)
    rows = [
        (
            ts, gw, p["id"], p["web_name"], p["team"], p["element_type"], p["now_cost"] / 10,
            float(p.get("selected_by_percent") or 0), p.get("status"),
            p.get("chance_of_playing_next_round"), float(p.get("form") or 0),
            p.get("total_points", 0), p.get("minutes", 0),
            float(p.get("expected_goals_per_90") or 0), float(p.get("expected_assists_per_90") or 0),
            float(p.get("ep_next") or 0), p.get("transfers_in_event", 0), p.get("transfers_out_event", 0),
        )
        for p in bootstrap["elements"]
    ]
    with _db() as con:
        con.executemany(f"INSERT OR REPLACE INTO player_snapshots VALUES ({','.join('?' * 18)})", rows)
    return ts


def _read_json(d: Path, name: str):
    plain, gz = d / f"{name}.json", d / f"{name}.json.gz"
    if plain.exists():
        return json.loads(plain.read_text())
    if gz.exists():
        with gzip.open(gz, "rt") as f:
            return json.load(f)
    raise FileNotFoundError(f"No {name}.json or {name}.json.gz in {d}")


def load_dir(path: Path) -> tuple[dict, list[dict]]:
    """Load bootstrap + fixtures from a directory holding {bootstrap,fixtures}.json[.gz]
    (tests/fixtures, or one data/raw/<ts>/ snapshot). A directory of snapshots (data/raw)
    resolves to its newest snapshot."""
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Offline path {path} is not a directory")
    if not any((path / f"bootstrap{ext}").exists() for ext in (".json", ".json.gz")):
        runs = sorted(p for p in path.iterdir() if p.is_dir() and any(p.glob("bootstrap.json*")))
        if not runs:
            raise FileNotFoundError(f"No bootstrap.json[.gz] in {path} or its subdirectories")
        path = runs[-1]
    return _read_json(path, "bootstrap"), _read_json(path, "fixtures")


def load_latest() -> tuple[dict, list[dict]]:
    raw = config.DATA_DIR / "raw"
    if not raw.is_dir() or not any(raw.iterdir()):
        raise FileNotFoundError("No snapshots yet — run `fplbot ingest` first")
    return load_dir(raw)


def record_decision(gw: int, plan: dict, executed: bool):
    with _db(_state_db_path()) as con:
        con.execute(
            "INSERT OR REPLACE INTO decisions VALUES (?,?,?,?)",
            (gw, datetime.now(timezone.utc).isoformat(), json.dumps(plan), int(executed)),
        )


def already_executed(gw: int) -> bool:
    with _db(_state_db_path()) as con:
        row = con.execute("SELECT executed FROM decisions WHERE gw=?", (gw,)).fetchone()
    return bool(row and row[0])
