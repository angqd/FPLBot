"""Score `project()` against FPL's frozen ep and a naive baseline, GW by GW.

Models (each a "variant" = overrides for the config's `model:` / `team_model:` / `eval:` blocks):
  model_pure    our model with model.blend_fpl_ep = 0
  model_blend   our model as configured (GW+1 blended with the baseline ep)
Baselines:
  ep            FPL's frozen pre-deadline ep_next for GW g (see history.py; horizon 3: ep x 3,
                FPL publishes no multi-GW ep)
  naive         points per team match so far x fixtures in the window
  ep_posthoc    DIAGNOSTIC ONLY, LEAKY: vaastav xP = ep_this recomputed after GW g with g's own
                points. Shown to quantify the leak; excluded from the decision set.

Sets per GW: `played` (minutes > 0 in the window) and `decision` (union of each model's top-N
projections, so non-players a model liked count against it). Metrics: MAE, RMSE, Spearman, plus
horizon-1 captain points (actual points of each model's top pick) and XI points (actual points of
the best-projected formation-legal XI, no budget).

The only contract with the model is `project(bootstrap, fixtures, start_gw, horizon, cfg_model,
team_model=...)` returning `xp_<gw>` columns (extra columns are ignored).
"""
from __future__ import annotations

import copy
import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from fplbot.eval import pit
from fplbot.eval.history import SeasonData, results_seasons
from fplbot.model import xpts
from fplbot.model.team_strength import TeamMappingError, fit_team_model

log = logging.getLogger(__name__)

EVAL_DEFAULTS = {
    "seasons": ["2026-27", "2025-26"],
    "current_season": "2026-27",
    "min_gw": 2,
    "horizons": [1, 3],
    "top_n": 100,
    "use_availability": True,
    "cache_max_age_hours": 12,
}
BASELINES = ("ep", "naive", "ep_posthoc")
DEFAULT_VARIANTS = {"model_pure": {"model": {"blend_fpl_ep": 0.0}}, "model_blend": {}}
XI_FORMATIONS = [(d, m, 10 - d - m) for d in range(3, 6) for m in range(2, 6) if 1 <= 10 - d - m <= 3]


def eval_cfg(cfg: dict) -> dict:
    return {**EVAL_DEFAULTS, **(cfg.get("eval") or {})}


def apply_overrides(cfg: dict, overrides: dict | None) -> dict:
    """Deep-merge {'model': {...}, 'team_model': {...}, 'eval': {...}} onto a copy of cfg."""
    out = copy.deepcopy(cfg)
    for block, vals in (overrides or {}).items():
        if isinstance(vals, dict):
            out[block] = {**(out.get(block) or {}), **vals}
        else:
            out[block] = vals
    return out


@dataclass
class BacktestResult:
    per_gw: pd.DataFrame            # one row per season/gw/horizon/set/model
    summary: pd.DataFrame           # pooled over GWs
    picks: pd.DataFrame             # horizon-1 captain / XI points per season/gw/model
    info: dict = field(default_factory=dict)


# ----------------------------------------------------------- predictions ----

def _played_matches(sd: SeasonData, g: int) -> pd.Series:
    past = sd.gw[sd.gw["gw"] < g]
    return past.groupby("element")["n_fixtures"].sum()


def naive_prediction(sd: SeasonData, bs: dict, fx: list[dict], g: int, horizon: int) -> pd.Series:
    """Points per team match so far (GWs < g) x the player's club fixtures in g..g+horizon-1."""
    window = set(range(g, g + horizon))
    nfx: dict[int, int] = {}
    for f in fx:
        if f["event"] in window:
            for t in (f["team_h"], f["team_a"]):
                nfx[t] = nfx.get(t, 0) + 1
    matches = _played_matches(sd, g)
    out = {}
    for e in bs["elements"]:
        m = float(matches.get(e["id"], 0))
        ppm = e["total_points"] / m if m > 0 else 0.0
        out[e["id"]] = ppm * nfx.get(e["team"], 0)
    return pd.Series(out, dtype=float)


def _xp_sum(df: pd.DataFrame, g: int, horizon: int) -> pd.Series:
    cols = [f"xp_{w}" for w in range(g, g + horizon) if f"xp_{w}" in df.columns]
    return pd.Series(df[cols].sum(axis=1).to_numpy(float), index=df["id"].to_numpy())


def actuals(sd: SeasonData, ids, g: int, horizon: int) -> pd.DataFrame:
    win = sd.gw[(sd.gw["gw"] >= g) & (sd.gw["gw"] < g + horizon)]
    a = win.groupby("element")[["total_points", "minutes"]].sum()
    return a.reindex(list(ids)).fillna(0.0)


# --------------------------------------------------------------- metrics ----

def _metrics(pred: pd.Series, act: pd.Series) -> dict:
    err = pred.to_numpy(float) - act.to_numpy(float)
    n = len(err)
    if n == 0:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "spearman": np.nan, "bias": np.nan}
    rho = spearmanr(pred, act).statistic if n > 2 and pred.nunique() > 1 and act.nunique() > 1 else np.nan
    return {"n": n, "mae": float(np.abs(err).mean()), "rmse": float(np.sqrt((err ** 2).mean())),
            "spearman": float(rho), "bias": float(err.mean())}


def best_xi(pred: pd.Series, pos: pd.Series) -> list[int]:
    """Formation-legal XI (1 GK, 3-5 DEF, 2-5 MID, 1-3 FWD) maximising projected points."""
    order = {p: pred[pos[pos == p].index].sort_values(ascending=False) for p in (1, 2, 3, 4)}
    best, best_ids = -math.inf, []
    for d, m, f in XI_FORMATIONS:
        picks = [order[1].index[:1], order[2].index[:d], order[3].index[:m], order[4].index[:f]]
        if sum(len(p) for p in picks) < 11:
            continue
        ids = [i for p in picks for i in p]
        tot = float(pred[ids].sum())
        if tot > best:
            best, best_ids = tot, ids
    return best_ids


# ------------------------------------------------------------------- run ----

def _fit_kwargs() -> dict:
    """unseen_as_promoted=True when the installed team model supports it (phase 2B), so early
    GWs with a promoted club that hasn't played yet don't raise TeamMappingError."""
    import inspect
    try:
        params = inspect.signature(fit_team_model).parameters
    except (TypeError, ValueError):
        return {}
    return {"unseen_as_promoted": True} if "unseen_as_promoted" in params else {}


def _team_model(bs, cfg, results, sd, g, cache):
    key = json.dumps(cfg.get("team_model") or {}, sort_keys=True, default=str)
    if key not in cache:
        res = pit.results_before(results, pit.deadline(sd, g))
        try:
            cache[key] = fit_team_model(bs, cfg, results=res, **_fit_kwargs())
        except TeamMappingError as e:
            log.warning("%s GW%s: team model can't be fitted (%s); GW skipped", sd.season, g, e)
            cache[key] = None
    return cache[key]


def predict_gw(sd: SeasonData, g: int, horizon: int, cfg: dict, results: pd.DataFrame,
               variants: dict[str, dict], _tm_cache: dict | None = None) -> pd.DataFrame | None:
    """Projections of every model/baseline for target GW g over `horizon` GWs, indexed by player id,
    columns = model names + 'pos'. Uses only data from GWs < g (see pit.py)."""
    fx = pit.fixtures_at(sd, g)
    boot_cache: dict[bool, dict] = {}
    tm_cache = {} if _tm_cache is None else _tm_cache
    out = {}
    for name, ov in variants.items():
        vcfg = apply_overrides(cfg, ov)
        sfe = bool(eval_cfg(vcfg)["use_availability"])
        if sfe not in boot_cache:
            boot_cache[sfe] = pit.bootstrap_at(sd, g, use_availability=sfe)
        bs = boot_cache[sfe]
        tm = _team_model(bs, vcfg, results, sd, g, tm_cache)
        if tm is None:
            return None
        df = xpts.project(bs, fx, g, horizon, vcfg["model"], team_model=tm)
        out[name] = _xp_sum(df, g, horizon)
    bs = next(iter(boot_cache.values())) if boot_cache else pit.bootstrap_at(sd, g)
    ep = pd.Series({e["id"]: float(e["ep_next"]) for e in bs["elements"] if e["ep_next"] is not None}, dtype=float)
    frame = pd.DataFrame(out)
    frame.index.name = "id"
    frame["ep"] = ep.reindex(frame.index).fillna(0.0) * horizon if not ep.empty else np.nan
    ph = sd.posthoc[sd.posthoc["gw"] == g].set_index("element")["ep"]
    frame["ep_posthoc"] = ph.reindex(frame.index).fillna(0.0) * horizon if not ph.empty else np.nan
    frame["naive"] = naive_prediction(sd, bs, fx, g, horizon).reindex(frame.index)
    frame["pos"] = pd.Series({e["id"]: e["element_type"] for e in bs["elements"]}).reindex(frame.index)
    return frame


def run_backtest(seasons: list[SeasonData], cfg: dict, results: dict[str, pd.DataFrame],
                 variants: dict[str, dict] | None = None, horizons=None, gws=None,
                 top_n: int | None = None) -> BacktestResult:
    """`results[season]` = football-data style results frame covering the season and the one before
    (filtered per GW to matches before the deadline). `gws` optionally limits target GWs
    (list, or {season: list})."""
    ec = eval_cfg(cfg)
    variants = dict(DEFAULT_VARIANTS if variants is None else variants)
    horizons = list(horizons or ec["horizons"])
    top_n = int(top_n or ec["top_n"])
    models = list(variants) + list(BASELINES)
    rows, pick_rows = [], []
    t0 = time.time()
    for sd in seasons:
        finished = sd.finished_gws
        last = max(finished) if finished else 0
        targets = gws.get(sd.season) if isinstance(gws, dict) else gws
        targets = [g for g in (targets or finished) if g >= ec["min_gw"] and g in sd.deadlines]
        for g in targets:
            tm_cache: dict = {}
            has_ep, has_ph, has_av = g in sd.ep_gws(), g in sd.posthoc_gws(), g in sd.avail_gws()
            for h in horizons:
                if g + h - 1 > last:
                    continue
                pred = predict_gw(sd, g, h, cfg, results[sd.season], variants, tm_cache)
                if pred is None:
                    break
                act = actuals(sd, pred.index, g, h)
                use = [m for m in models if (m != "ep" or has_ep) and (m != "ep_posthoc" or has_ph)]
                decision = set()
                for m in use:
                    if m != "ep_posthoc":  # leaky diagnostic: must not shape the decision set
                        decision |= set(pred[m].nlargest(top_n).index)
                sets = {"played": act.index[act["minutes"] > 0], "decision": sorted(decision)}
                for set_name, ids in sets.items():
                    for m in use:
                        r = _metrics(pred.loc[ids, m], act.loc[ids, "total_points"])
                        rows.append({"season": sd.season, "gw": g, "horizon": h, "set": set_name,
                                     "model": m, "ep_source": sd.ep_source(g), "has_avail": has_av, **r})
                if h == 1:
                    for m in use:
                        cap = pred[m].idxmax()
                        xi = best_xi(pred[m], pred["pos"])
                        pick_rows.append({"season": sd.season, "gw": g, "model": m, "captain": int(cap),
                                          "captain_pts": float(act.loc[cap, "total_points"]),
                                          "xi_pts": float(act.loc[xi, "total_points"].sum()),
                                          "has_avail": has_av})
            log.info("%s GW%s done (%.1fs)", sd.season, g, time.time() - t0)
    per_gw = pd.DataFrame(rows)
    picks = pd.DataFrame(pick_rows)
    summary = summarize(per_gw, picks)
    return BacktestResult(per_gw, summary, picks, {"runtime_s": round(time.time() - t0, 1),
                                                   "variants": variants, "top_n": top_n})


def summarize(per_gw: pd.DataFrame, picks: pd.DataFrame) -> pd.DataFrame:
    """Pool per-GW metrics. scope 'all_gws' = every evaluated GW; 'avail_gws' = GWs where each
    player's chance of playing at the deadline was recovered (our model sees injury flags there,
    like the live bot does); 'season:<s>' = one season. MAE/RMSE are n-weighted (pooled); Spearman and captain/XI points
    are per-GW means."""
    if per_gw.empty:
        return pd.DataFrame()
    out = []
    scopes = {"all_gws": (per_gw, picks),
              "avail_gws": (per_gw[per_gw["has_avail"].astype(bool)],
                            picks[picks["has_avail"].astype(bool)] if not picks.empty else picks)}
    for season in sorted(per_gw["season"].unique()):
        scopes[f"season:{season}"] = (per_gw[per_gw["season"] == season],
                                      picks[picks["season"] == season] if not picks.empty else picks)
    for scope, (df, pk) in scopes.items():
        for (h, s, m), d in df.groupby(["horizon", "set", "model"]):
            n = d["n"].sum()
            row = {"scope": scope, "horizon": h, "set": s, "model": m, "gws": len(d), "n": int(n),
                   "mae": float((d["mae"] * d["n"]).sum() / n),
                   "rmse": float(np.sqrt(((d["rmse"] ** 2) * d["n"]).sum() / n)),
                   "spearman": float(d["spearman"].mean()), "bias": float((d["bias"] * d["n"]).sum() / n)}
            if h == 1 and not pk.empty:
                p = pk[pk["model"] == m]
                row["captain_pts"] = float(p["captain_pts"].mean()) if len(p) else np.nan
                row["xi_pts"] = float(p["xi_pts"].mean()) if len(p) else np.nan
            out.append(row)
    return pd.DataFrame(out)


def format_table(summary: pd.DataFrame, scope: str = "all_gws") -> str:
    df = summary[summary["scope"] == scope].copy()
    if df.empty:
        return f"(no rows for scope {scope})"
    cols = ["horizon", "set", "model", "gws", "n", "mae", "rmse", "spearman", "bias", "captain_pts", "xi_pts"]
    df = df[[c for c in cols if c in df.columns]].sort_values(["horizon", "set", "mae"])
    return df.to_string(index=False, float_format=lambda x: f"{x:.3f}", na_rep="-")


# ---------------------------------------------------------------- loading ----

def load_results_for(season: str, cfg: dict, offline: bool = False, cache_dir: Path | None = None) -> pd.DataFrame:
    from fplbot.model.team_strength import load_results
    tm = {**(cfg.get("team_model") or {}), "seasons": results_seasons(season)}
    return load_results({"team_model": tm}, offline=offline, cache_dir=cache_dir)


def load_all(cfg: dict, seasons=None, offline: bool = False, data_dir: Path | None = None):
    """-> (list[SeasonData], {season: results}) for the configured eval seasons."""
    from fplbot.eval.history import load_season
    ec = eval_cfg(cfg)
    if data_dir is None:
        from fplbot.config import DATA_DIR
        data_dir = DATA_DIR
    sds, res = [], {}
    for s in seasons or ec["seasons"]:
        sds.append(load_season(s, ec, offline=offline, data_dir=data_dir))
        res[s] = load_results_for(s, cfg, offline=offline, cache_dir=Path(data_dir) / "cache")
    return sds, res


def evaluate_configs(overrides: list[dict], seasons: list[SeasonData], cfg: dict,
                     results: dict[str, pd.DataFrame], horizons=(1,), gws=None,
                     set_name: str = "decision", scope: str = "all_gws") -> list[dict]:
    """Tuning hook. Each override is {'model': {...}, 'team_model': {...}, 'eval': {...}} (any subset).
    Returns one dict per override: the override plus its pooled metrics (per horizon) on `set_name`.

        evaluate_configs([{"model": {"blend_fpl_ep": w}} for w in (0, .2, .35, .5)], sds, cfg, res)
    """
    variants = {f"cfg{i}": ov for i, ov in enumerate(overrides)}
    r = run_backtest(seasons, cfg, results, variants=variants, horizons=horizons, gws=gws)
    out = []
    for i, ov in enumerate(overrides):
        s = r.summary[(r.summary["model"] == f"cfg{i}") & (r.summary["set"] == set_name)
                      & (r.summary["scope"] == scope)]
        out.append({"overrides": ov, "metrics": {int(row["horizon"]): {k: row[k] for k in (
            "mae", "rmse", "spearman", "bias", "captain_pts", "xi_pts", "gws", "n") if k in row and pd.notna(row[k])}
            for _, row in s.iterrows()}})
    return out


def save(result: BacktestResult, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%dT%H%M%S")
    paths = {"per_gw": out_dir / f"backtest_{ts}_per_gw.csv", "picks": out_dir / f"backtest_{ts}_picks.csv",
             "summary_csv": out_dir / f"backtest_{ts}_summary.csv", "summary_json": out_dir / f"backtest_{ts}_summary.json"}
    result.per_gw.to_csv(paths["per_gw"], index=False)
    result.picks.to_csv(paths["picks"], index=False)
    result.summary.to_csv(paths["summary_csv"], index=False)
    paths["summary_json"].write_text(json.dumps({"info": result.info, "summary": result.summary.to_dict("records")},
                                                indent=1, default=str))
    return paths
