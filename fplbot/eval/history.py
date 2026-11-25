"""Per-player per-gameweek history plus a frozen pre-deadline FPL ep baseline for each GW.

Sources (all public, cached under DATA_DIR/cache, which is gitignored):

* vaastav/Fantasy-Premier-League `data/<season>/gws/merged_gw.csv`: one row per player per
  fixture with the GW's stats and `value` (price at that GW).
* FPL `event/{gw}/live/`: tops up finished GWs of the current season vaastav hasn't published.
* Our own ingest snapshots (DATA_DIR/raw/<ts>/bootstrap.json.gz).

The ep baseline. In 2026/27 FPL's `ep_next` is exactly `form` x chance of playing (checked on a
live snapshot: ep_next == form for all 340 available players with form > 0), and `form` is the
mean points per GW over GWs whose deadline is in the last 30 days (reconstructed exactly from
history for all 674 players). vaastav's `xP` column is `ep_this` scraped *after* the GW, and
FPL recomputes it with the GW's own points: xP(g) equals the mean over a window that ENDS at g
for ~85-93% of players. So xP is LEAKY and is not used as the baseline. Instead, per GW g:

  1. `snapshot_ep_next_pre_deadline`: ep_next from our latest snapshot taken before g's
     deadline (truly frozen), else
  2. `reconstructed_form`: form at g's deadline (GWs < g only) x availability x [club has a
     fixture in g]. Availability = chance_of_playing_this_round at the deadline, recovered as
     xP(g) / form-including-g (FPL freezes that chance at the deadline; e.g. players suspended
     after the GW11 deadline kept ep_this > 0), snapped to quarters; 1 when unknown.

The leaky post-hoc xP is kept (`SeasonData.posthoc`) only as a diagnostic.
"""
from __future__ import annotations

import gzip
import io
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

VAASTAV_BASE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
POSITIONS = {"GK": 1, "GKP": 1, "DEF": 2, "MID": 3, "FWD": 4}  # AM (assistant manager) dropped
STAT_COLS = [
    "minutes", "starts", "goals_scored", "assists", "clean_sheets", "goals_conceded", "own_goals",
    "penalties_saved", "penalties_missed", "yellow_cards", "red_cards", "saves", "bonus", "bps",
    "defensive_contribution", "expected_goals", "expected_assists", "expected_goal_involvements",
    "expected_goals_conceded", "total_points",
]
GW_COLUMNS = ["element", "gw", "team", "element_type", "web_name", "value", "n_fixtures", *STAT_COLS]
EP_COLUMNS = ["element", "gw", "ep", "source"]
FORM_DAYS = 30
MIN_FORM_MATCH = 0.75  # trust ep_this-derived availability only when ep_this == form this often
AVAIL_COLUMNS = ["element", "gw", "chance"]


@dataclass
class SeasonData:
    """Everything a backtest needs for one season. `gw` has one row per (element, gw) with
    GW_COLUMNS; `ep` is the frozen baseline (EP_COLUMNS); `avail` the chance of playing at each
    deadline where recoverable (AVAIL_COLUMNS, 0..1); `posthoc` the leaky post-GW ep_this
    (diagnostic only); deadlines are ISO UTC."""
    season: str
    gw: pd.DataFrame
    ep: pd.DataFrame
    teams: list[dict]
    fixtures: list[dict]
    deadlines: dict[int, str]
    avail: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=AVAIL_COLUMNS))
    posthoc: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=EP_COLUMNS))

    @property
    def finished_gws(self) -> list[int]:
        return sorted(int(g) for g in self.gw["gw"].unique())

    def ep_gws(self) -> set[int]:
        return set(int(g) for g in self.ep["gw"].unique())

    def avail_gws(self) -> set[int]:
        return set(int(g) for g in self.avail["gw"].unique())

    def posthoc_gws(self) -> set[int]:
        return set(int(g) for g in self.posthoc["gw"].unique())

    def ep_source(self, gw: int) -> str | None:
        s = self.ep.loc[self.ep["gw"] == gw, "source"]
        return None if s.empty else str(s.iloc[0])


def season_start_year(season: str) -> int:
    return int(season[:4])


def results_seasons(season: str) -> list[str]:
    """'2026-27' -> football-data codes for the previous and this season: ['2526', '2627']."""
    y = season_start_year(season)
    return [f"{(y - 1) % 100:02d}{y % 100:02d}", f"{y % 100:02d}{(y + 1) % 100:02d}"]


# ------------------------------------------------------------- download ----

def _cached_text(url: str, path: Path, max_age_hours: float | None, offline: bool) -> str | None:
    """Return cached text (fresh enough, or any cache when offline/download fails), else download.
    max_age_hours=None means the cache never expires. Returns None on a 404."""
    fresh = path.exists() and (max_age_hours is None or time.time() - path.stat().st_mtime < max_age_hours * 3600)
    if fresh or (offline and path.exists()):
        return path.read_text()
    if offline:
        raise FileNotFoundError(f"offline and no cache at {path}")
    import httpx
    try:
        r = httpx.get(url, timeout=60, follow_redirects=True, headers={"User-Agent": "fplbot-eval/0.1"})
        if r.status_code == 404:
            return None
        r.raise_for_status()
    except Exception as e:  # noqa: BLE001
        if path.exists():
            log.warning("download of %s failed (%s); using stale cache", url, e)
            return path.read_text()
        raise
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(r.text)
    return r.text


def load_vaastav(season: str, cache_dir: Path, max_age_hours: float | None, offline: bool = False) -> dict:
    """merged_gw / teams / fixtures / players_raw for one season as DataFrames (players_raw optional)."""
    out = {}
    for key, rel in (("merged_gw", "gws/merged_gw.csv"), ("teams", "teams.csv"),
                     ("fixtures", "fixtures.csv"), ("players_raw", "players_raw.csv")):
        path = cache_dir / "vaastav" / season / rel.replace("/", "_")
        try:
            text = _cached_text(f"{VAASTAV_BASE}/{season}/{rel}", path, max_age_hours, offline)
        except FileNotFoundError:
            if key == "players_raw":
                text = None
            else:
                raise
        out[key] = None if text is None else pd.read_csv(io.StringIO(text), low_memory=False)
    if out["merged_gw"] is None:
        raise FileNotFoundError(f"vaastav has no merged_gw.csv for {season}")
    return out


# ------------------------------------------------------------ transform ----

def aggregate_fixture_rows(rows: pd.DataFrame, teams: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """vaastav merged_gw rows (one per player per fixture) -> (gw frame, post-hoc xP frame).
    Double GWs are summed; value/xP are per GW so the first row is kept."""
    df = rows.copy()
    df["gw"] = pd.to_numeric(df["GW"] if "GW" in df else df["round"]).astype(int)
    df["element_type"] = df["position"].map(POSITIONS)
    df = df.dropna(subset=["element_type"])
    if not pd.api.types.is_numeric_dtype(df["team"]):
        name_to_id = dict(zip(teams["name"], teams["id"]))
        df["team"] = df["team"].map(name_to_id)
    df = df.dropna(subset=["team"])
    for c in STAT_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0) if c in df else 0.0
    df["xP"] = pd.to_numeric(df.get("xP", 0.0), errors="coerce").fillna(0.0)
    if "kickoff_time" in df:
        df = df.sort_values("kickoff_time", kind="stable")
    g = df.groupby(["element", "gw"], sort=True)
    agg = g[STAT_COLS].sum()
    agg["team"] = g["team"].last().astype(int)
    agg["element_type"] = g["element_type"].last().astype(int)
    agg["web_name"] = g["name"].first() if "name" in df else ""
    agg["value"] = g["value"].first().astype(int)
    agg["n_fixtures"] = g["fixture"].nunique() if "fixture" in df else g.size()
    agg["xP"] = g["xP"].first()
    agg = agg.reset_index()
    # a GW whose xP file was missing in the scrape has xP == 0 for everyone: no baseline
    has_ep = agg.groupby("gw")["xP"].transform(lambda s: (s != 0).any())
    ep = agg.loc[has_ep, ["element", "gw", "xP"]].rename(columns={"xP": "ep"})
    ep["source"] = "vaastav_xP_posthoc"
    return agg[GW_COLUMNS], ep[EP_COLUMNS].reset_index(drop=True)


def live_to_rows(live: dict, gw: int, elements: list[dict], fixtures: list[dict],
                 last_value: dict[int, int] | None = None) -> pd.DataFrame:
    """FPL event/{gw}/live/ payload -> gw frame rows. Team/position come from `elements`
    (a bootstrap-static elements list)."""
    meta = {e["id"]: e for e in elements}
    nfx: dict[int, int] = {}
    for fx in fixtures:
        if fx.get("event") == gw:
            for t in (fx["team_h"], fx["team_a"]):
                nfx[t] = nfx.get(t, 0) + 1
    last_value = last_value or {}
    rows = []
    for el in live["elements"]:
        m = meta.get(el["id"])
        if m is None:
            continue
        s = el["stats"]
        row = {"element": el["id"], "gw": gw, "team": m["team"], "element_type": m["element_type"],
               "web_name": m.get("web_name", ""), "value": int(last_value.get(el["id"], m["now_cost"])),
               "n_fixtures": nfx.get(m["team"], 0)}
        for c in STAT_COLS:
            row[c] = float(s.get(c) or 0)
        rows.append(row)
    return pd.DataFrame(rows, columns=GW_COLUMNS)


def _deadlines_from_fixtures(fixtures: pd.DataFrame) -> dict[int, str]:
    """FPL deadlines are 90 minutes before the GW's first kickoff."""
    fx = fixtures.dropna(subset=["event", "kickoff_time"])
    first = pd.to_datetime(fx["kickoff_time"], utc=True).groupby(fx["event"].astype(int)).min()
    return {int(g): (t - timedelta(minutes=90)).strftime("%Y-%m-%dT%H:%M:%SZ") for g, t in first.items()}


def _fixture_dicts(fixtures: pd.DataFrame) -> list[dict]:
    out = []
    for r in fixtures.itertuples(index=False):
        ev = getattr(r, "event")
        out.append({"id": int(r.id), "event": None if pd.isna(ev) else int(ev),
                    "team_h": int(r.team_h), "team_a": int(r.team_a),
                    "kickoff_time": None if pd.isna(r.kickoff_time) else str(r.kickoff_time)})
    return out


def _team_dicts(teams: pd.DataFrame) -> list[dict]:
    out = []
    for r in teams.to_dict("records"):
        t = {"id": int(r["id"]), "name": str(r["name"]), "short_name": str(r.get("short_name", "")),
             "code": int(r["code"]) if "code" in r and not pd.isna(r["code"]) else None, "strength": None}
        for k in ("strength_overall_home", "strength_overall_away"):
            t[k] = int(r[k]) if k in r and not pd.isna(r[k]) else 3
        # 0 like the live API, so project() must use the fitted team model
        for k in ("strength_attack_home", "strength_attack_away", "strength_defence_home", "strength_defence_away"):
            t[k] = 0
        out.append(t)
    return out


# ------------------------------------------------------------ snapshots ----

def _snapshot_ts(d: Path) -> datetime | None:
    try:
        return datetime.strptime(d.name, "%Y-%m-%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def iter_snapshots(raw_dir: Path, season: str):
    """Yield (timestamp, bootstrap) for this season's ingest snapshots, oldest first."""
    if not raw_dir.is_dir():
        return
    for d in sorted(raw_dir.iterdir()):
        ts = _snapshot_ts(d)
        f = d / "bootstrap.json.gz"
        if ts is None or not f.exists():
            continue
        with gzip.open(f, "rt") as fh:
            bs = json.load(fh)
        first = min((e["deadline_time"] for e in bs.get("events", []) if e.get("deadline_time")), default=None)
        if first and _parse_iso(first).year == season_start_year(season):
            yield ts, bs


def snapshot_eps(snapshots) -> tuple[pd.DataFrame, pd.DataFrame]:
    """-> (pre-deadline ep_next rows, post-deadline ep_this rows) from ingest snapshots.
    ep_next of the next GW counts if the snapshot was taken before that GW's deadline (latest such
    snapshot wins); ep_this of the current GW if taken after its deadline (earliest wins)."""
    best: dict[tuple[int, str], tuple[datetime, dict]] = {}
    for ts, bs in snapshots:
        for ev in bs["events"]:
            dl = _parse_iso(ev["deadline_time"]) if ev.get("deadline_time") else None
            if dl is None:
                continue
            if ev.get("is_next") and ts < dl:
                key, fld, later_wins = (ev["id"], "snapshot_ep_next_pre_deadline"), "ep_next", True
            elif ev.get("is_current") and ts >= dl:
                key, fld, later_wins = (ev["id"], "snapshot_ep_this_posthoc"), "ep_this", False
            else:
                continue
            prev = best.get(key)
            if prev is None or (ts > prev[0] if later_wins else ts < prev[0]):
                best[key] = (ts, {e["id"]: e.get(fld) for e in bs["elements"]})
    rows = [{"element": pid, "gw": gw, "ep": float(v), "source": src}
            for (gw, src), (_, eps) in best.items() for pid, v in eps.items() if v is not None]
    df = pd.DataFrame(rows, columns=EP_COLUMNS)
    pre = df["source"] == "snapshot_ep_next_pre_deadline"
    return df[pre].reset_index(drop=True), df[~pre].reset_index(drop=True)


def _points_wide(gw: pd.DataFrame) -> pd.DataFrame:
    return gw.pivot_table(index="element", columns="gw", values="total_points", aggfunc="sum").fillna(0.0)


def form_at(pts: pd.DataFrame, deadlines: dict[int, str], g: int, days: int = FORM_DAYS) -> pd.Series:
    """FPL `form` as it stood at GW g's deadline: mean points per GW over GWs k < g whose deadline
    is within `days` before g's deadline. Uses only columns k < g of `pts`."""
    dg = _parse_iso(deadlines[g])
    window = [k for k in pts.columns if k < g and k in deadlines
              and dg - timedelta(days=days) <= _parse_iso(deadlines[k]) < dg]
    if not window:
        return pd.Series(0.0, index=pts.index)
    return (pts[window].sum(axis=1) / len(window)).round(1)


def availability_from_posthoc(pts: pd.DataFrame, posthoc_g: pd.Series, g: int) -> tuple[pd.Series, float]:
    """Chance of playing at GW g's deadline (0..1) from the post-hoc ep_this(g) =
    form-including-g x chance_this_round. The form window ending at g is the one that best
    reproduces ep_this; ratios are snapped to quarters and only trusted when close to one
    (else 1). Players with ep_this 0 but positive form-including-g get chance 0.
    Returns (chance, match_rate): the share of players whose ep_this equals that form exactly."""
    idx = posthoc_g.index.intersection(pts.index)
    ep = posthoc_g[idx].astype(float)
    best, form = -1.0, None
    for start in range(max(1, g - 6), g + 1):
        cols = [c for c in range(start, g + 1) if c in pts.columns]
        if not cols:
            continue
        f = pts.loc[idx, cols].mean(axis=1).round(1)
        m = (f > 0) | (ep > 0)
        rate = float((ep[m] == f[m]).mean()) if m.any() else 0.0
        if rate > best:
            best, form = rate, f
    chance = pd.Series(1.0, index=idx)
    if form is None:
        return chance, 0.0
    pos = form > 0
    r = (ep[pos] / form[pos]).clip(0, 1)
    q = (r * 4).round() / 4
    trusted = (r * 4 - q * 4).abs() < 0.12
    chance[q.index[trusted]] = q[trusted].to_numpy(dtype=float)
    return chance, best


def reconstruct_baseline(gw: pd.DataFrame, deadlines: dict[int, str], fixtures: list[dict],
                         posthoc: pd.DataFrame, pre_deadline: pd.DataFrame | None = None,
                         first_gw: int = 2, min_match: float = MIN_FORM_MATCH) -> tuple[pd.DataFrame, pd.DataFrame]:
    """-> (ep baseline EP_COLUMNS, availability AVAIL_COLUMNS) for every finished GW >= first_gw.
    Availability is only recovered for GWs where ep_this == form for >= `min_match` of players
    (true for 2026/27; in 2025/26 FPL's ep was not plain form, so it's skipped there)."""
    pts = _points_wide(gw)
    pre = pre_deadline if pre_deadline is not None else pd.DataFrame(columns=EP_COLUMNS)
    pre_gws = set(pre["gw"].unique())
    has_fx: dict[int, set[int]] = {}
    for fx in fixtures:
        if fx.get("event") is not None:
            has_fx.setdefault(fx["event"], set()).update((fx["team_h"], fx["team_a"]))
    eps, avails = [pre], []
    for g in sorted(set(gw["gw"].unique())):
        if g < first_gw or g not in deadlines:
            continue
        ph = posthoc[posthoc["gw"] == g].set_index("element")["ep"]
        chance = None
        if not ph.empty:
            chance, rate = availability_from_posthoc(pts[[c for c in pts.columns if c <= g]], ph, g)
            if rate < min_match:
                log.info("GW%s: ep_this matches form for only %.0f%% of players; availability not recovered", g, 100 * rate)
                chance = None
        if chance is not None:
            avails.append(pd.DataFrame({"element": chance.index, "gw": g, "chance": chance.to_numpy()}))
        if g in pre_gws:
            continue
        rows_g = gw[gw["gw"] <= g].sort_values("gw").groupby("element")["team"].last()
        form = form_at(pts[[c for c in pts.columns if c < g]], deadlines, g).reindex(rows_g.index).fillna(0.0)
        c = chance.reindex(rows_g.index).fillna(1.0) if chance is not None else 1.0
        plays = rows_g.map(lambda t: t in has_fx.get(g, set())).astype(float)
        eps.append(pd.DataFrame({"element": rows_g.index, "gw": g, "ep": (form * c * plays).round(1).to_numpy(),
                                 "source": "reconstructed_form"}))
    ep = pd.concat([e for e in eps if not e.empty], ignore_index=True) if any(not e.empty for e in eps) \
        else pd.DataFrame(columns=EP_COLUMNS)
    av = pd.concat(avails, ignore_index=True) if avails else pd.DataFrame(columns=AVAIL_COLUMNS)
    return ep[EP_COLUMNS], av[AVAIL_COLUMNS]


# ----------------------------------------------------------------- load ----

def load_season(season: str, cfg: dict | None = None, offline: bool = False,
                data_dir: Path | None = None, client=None) -> SeasonData:
    """Build SeasonData from vaastav (+ FPL live top-up and snapshot eps for the current season)."""
    cfg = cfg or {}
    if data_dir is None:
        from fplbot.config import DATA_DIR
        data_dir = DATA_DIR
    data_dir = Path(data_dir)
    cache = data_dir / "cache"
    current = season == cfg.get("current_season")
    max_age = cfg.get("cache_max_age_hours", 12) if current else None
    v = load_vaastav(season, cache, max_age, offline)
    gw, posthoc = aggregate_fixture_rows(v["merged_gw"], v["teams"])
    if v["players_raw"] is not None and "web_name" in v["players_raw"]:
        names = dict(zip(v["players_raw"]["id"], v["players_raw"]["web_name"]))
        gw["web_name"] = gw["element"].map(names).fillna(gw["web_name"])
    fixtures = _fixture_dicts(v["fixtures"])
    deadlines = _deadlines_from_fixtures(v["fixtures"])
    snaps = list(iter_snapshots(data_dir / "raw", season))
    pre, snap_posthoc = snapshot_eps(snaps)
    # vaastav post-hoc xP wins over our snapshot's for the same GW (one source per GW)
    snap_posthoc = snap_posthoc[~snap_posthoc["gw"].isin(set(posthoc["gw"]))]
    posthoc = pd.concat([posthoc, snap_posthoc], ignore_index=True)

    if current:
        latest = snaps[-1][1] if snaps else None
        if latest is None and not offline:
            from fplbot.data.fpl_client import FPLClient
            client = client or FPLClient()
            latest = client.bootstrap()
        if latest is not None:
            deadlines.update({e["id"]: e["deadline_time"] for e in latest["events"] if e.get("deadline_time")})
            done = [e["id"] for e in latest["events"] if e.get("finished") and e.get("data_checked")]
            missing = [g for g in done if g not in set(gw["gw"])]
            last_value = gw.sort_values("gw").groupby("element")["value"].last().to_dict()
            extra = []
            for g in missing:
                path = cache / "fpl_live" / season / f"gw{g}.json"
                if path.exists():
                    live = json.loads(path.read_text())
                elif offline:
                    log.warning("offline: no cached event/%s/live; GW%s skipped", g, g)
                    continue
                else:
                    from fplbot.data.fpl_client import FPLClient
                    client = client or FPLClient()
                    live = client.live(g)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(live))
                extra.append(live_to_rows(live, g, latest["elements"], fixtures, last_value))
                log.info("GW%s topped up from FPL event/live", g)
            if extra:
                gw = pd.concat([gw, *extra], ignore_index=True)
    gw = gw.reset_index(drop=True)
    ep, avail = reconstruct_baseline(gw, deadlines, fixtures, posthoc, pre)
    return SeasonData(season, gw, ep, _team_dicts(v["teams"]), fixtures, deadlines, avail, posthoc)
