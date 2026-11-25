"""Estimate the promoted-club prior from past Premier League seasons.

For each season s in the history, fit ratings on season s alone (no time decay, a weak ridge so
the result is close to the unshrunk MLE, promoted prior off, same target weights as production;
seasons without xG renormalise over goals + odds), re-centred so that season's 20 clubs average
0. The clubs in s that weren't in s−1 are that season's promoted clubs; the prior is the mean of
their first-season attack and defence ratings.

    python -m fplbot.model.promoted_prior            # downloads/caches E0 1516..2526
    python -m fplbot.model.promoted_prior --offline  # cache only
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from fplbot.model.team_strength import _cfg, fit_ratings, load_results


def season_codes(first: int, last: int) -> list[str]:
    """season_codes(2015, 2025) -> ['1516', ..., '2526'] (start years inclusive)."""
    return [f"{y % 100:02d}{(y + 1) % 100:02d}" for y in range(first, last + 1)]


def promoted_ratings(results: pd.DataFrame, cfg=None) -> pd.DataFrame:
    """One row per promoted club-season: season, team, att, def, n."""
    c = _cfg(cfg)
    hist_cfg = {**c, "xi": 0.0, "ridge_sigma": 1.0, "promoted_prior": {"enabled": False}}
    rows = []
    seasons = sorted(set(results["season"]))
    for prev, s in zip(seasons, seasons[1:]):
        if s != prev + 1:
            continue
        cur, before = results[results["season"] == s], results[results["season"] == prev]
        promoted = (set(cur["home"]) | set(cur["away"])) - (set(before["home"]) | set(before["away"]))
        r = fit_ratings(cur, hist_cfg, promoted=())
        for t in sorted(promoted):
            rows.append({"season": s, "team": t, "att": r.att[t], "def": r.dfn[t], "n": r.n_matches[t]})
    return pd.DataFrame(rows)


def estimate(results: pd.DataFrame, cfg=None) -> dict:
    pr = promoted_ratings(results, cfg)
    return {"att": float(pr["att"].mean()), "def": float(pr["def"].mean()),
            "att_sd": float(pr["att"].std()), "def_sd": float(pr["def"].std()),
            "n_clubs": len(pr), "seasons": sorted(set(pr["season"])), "table": pr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=2015)
    ap.add_argument("--last", type=int, default=2025)
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()
    c = _cfg(None)
    # historical CSVs never change: cache them indefinitely
    res = load_results({**c, "seasons": season_codes(a.first, a.last), "cache_max_age_hours": 1e9},
                       offline=a.offline)
    est = estimate(res, c)
    with pd.option_context("display.width", 120, "display.max_rows", 100):
        print(est["table"].round(3).to_string(index=False))
    print(f"\npromoted clubs: {est['n_clubs']} over seasons {est['seasons'][0]}..{est['seasons'][-1]}")
    print(f"prior att {est['att']:+.3f} (sd {est['att_sd']:.3f}), def {est['def']:+.3f} (sd {est['def_sd']:.3f})")
    print(f"mean exp(att - def) multiplier on goal diff scale: {np.exp(est['att']):.2f}x goals for, "
          f"{np.exp(-est['def']):.2f}x goals against")


if __name__ == "__main__":
    main()
