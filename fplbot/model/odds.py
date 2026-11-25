"""Betting odds -> market-implied expected goals.

Each match's 1X2 and over/under-2.5 prices are de-vigged (proportional normalisation) and turned
into two independent-Poisson means (λ_home, λ_away):

1. The total T = λ_home + λ_away is set so that P(total goals ≤ 2 | Poisson(T)) equals the
   de-vigged P(under 2.5). Under independence the total is Poisson(T), so this is exact.
2. The split s = λ_home / T is set so that P(H) − P(A) under the two Poissons equals the
   de-vigged P(H) − P(A). The difference is monotone in s, so both solves are bisections.

Independent Poissons can't match P(H), P(D), P(A) and P(over) all at once (real football has a
few more draws), so we match the two quantities that pin down the means: the total and the
home/away balance. Everything is vectorised over matches (a season solves in milliseconds).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Column triples/pairs in order of preference. Closing odds are the most efficient public
# predictor and are still known before kick-off; averages beat a single bookmaker. Older seasons
# (pre 2019/20) use Betbrain names (BbAv*). Picked per row, so a missing value falls through.
ODDS_1X2 = [("AvgCH", "AvgCD", "AvgCA"), ("AvgH", "AvgD", "AvgA"), ("BbAvH", "BbAvD", "BbAvA"),
            ("PSCH", "PSCD", "PSCA"), ("PSH", "PSD", "PSA"),
            ("B365CH", "B365CD", "B365CA"), ("B365H", "B365D", "B365A")]
ODDS_OU25 = [("AvgC>2.5", "AvgC<2.5"), ("Avg>2.5", "Avg<2.5"), ("BbAv>2.5", "BbAv<2.5"),
             ("PC>2.5", "PC<2.5"), ("P>2.5", "P<2.5"),
             ("B365C>2.5", "B365C<2.5"), ("B365>2.5", "B365<2.5")]

MAX_GOALS = 15


def _pick(df: pd.DataFrame, groups) -> np.ndarray:
    """First complete group of odds columns per row -> (n, k) array of decimal odds (NaN if none)."""
    k = len(groups[0])
    out = np.full((len(df), k), np.nan)
    for cols in groups:
        if not set(cols) <= set(df.columns):
            continue
        vals = np.column_stack([pd.to_numeric(df[c], errors="coerce").to_numpy(float) for c in cols])
        ok = np.all(vals > 1.0, axis=1) & np.isnan(out).any(axis=1)
        out[ok] = vals[ok]
    return out


def devig(odds: np.ndarray) -> np.ndarray:
    """Decimal odds (n, k) -> probabilities with the bookmaker margin removed proportionally."""
    inv = 1.0 / np.asarray(odds, float)
    return inv / inv.sum(axis=1, keepdims=True)


def market_probs(df: pd.DataFrame) -> pd.DataFrame:
    """football-data rows -> de-vigged columns ph, pd, pa, pover (NaN where odds are missing)."""
    p1x2 = devig(_pick(df, ODDS_1X2))
    pou = devig(_pick(df, ODDS_OU25))
    return pd.DataFrame({"ph": p1x2[:, 0], "pd": p1x2[:, 1], "pa": p1x2[:, 2], "pover": pou[:, 0]},
                        index=df.index)


def _pmf(lam: np.ndarray) -> np.ndarray:
    """Poisson pmf for k = 0..MAX_GOALS, shape (n, MAX_GOALS+1)."""
    k = np.arange(MAX_GOALS + 1)
    logfact = np.cumsum(np.log(np.maximum(k, 1)))
    return np.exp(k * np.log(lam[:, None]) - lam[:, None] - logfact)


def outcome_probs(lh, la) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Independent-Poisson P(H), P(D), P(A), P(over 2.5) for arrays of means."""
    lh, la = np.atleast_1d(np.asarray(lh, float)), np.atleast_1d(np.asarray(la, float))
    ph, pa = _pmf(lh), _pmf(la)
    joint = ph[:, :, None] * pa[:, None, :]
    tri = np.tril(np.ones((MAX_GOALS + 1,) * 2), -1).astype(bool)  # i > j: home win
    p_h = joint[:, tri].sum(1)
    p_a = joint[:, tri.T].sum(1)
    p_d = np.einsum("nii->n", joint)
    t = lh + la
    p_over = 1.0 - np.exp(-t) * (1 + t + t * t / 2)
    return p_h, p_d, p_a, p_over


def implied_lambdas(ph, pa, pover, iters: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """De-vigged P(home win), P(away win), P(over 2.5) -> (λ_home, λ_away). NaN in, NaN out."""
    ph, pa, pover = (np.atleast_1d(np.asarray(x, float)) for x in (ph, pa, pover))
    ok = ~(np.isnan(ph) | np.isnan(pa) | np.isnan(pover))
    lh, la = np.full(ph.shape, np.nan), np.full(ph.shape, np.nan)
    if not ok.any():
        return lh, la
    target_under = np.clip(1.0 - pover[ok], 1e-4, 1 - 1e-4)
    lo, hi = np.full(target_under.shape, 0.05), np.full(target_under.shape, 10.0)
    for _ in range(iters):  # P(under | T) is decreasing in T
        mid = 0.5 * (lo + hi)
        under = np.exp(-mid) * (1 + mid + mid * mid / 2)
        lo, hi = np.where(under > target_under, mid, lo), np.where(under > target_under, hi, mid)
    total = 0.5 * (lo + hi)

    target_diff = ph[ok] - pa[ok]
    lo, hi = np.full(total.shape, 0.01), np.full(total.shape, 0.99)
    for _ in range(iters // 2):  # P(H) − P(A) is increasing in the home share
        mid = 0.5 * (lo + hi)
        p_h, _, p_a, _ = outcome_probs(mid * total, (1 - mid) * total)
        below = (p_h - p_a) < target_diff
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    s = 0.5 * (lo + hi)
    lh[ok], la[ok] = s * total, (1 - s) * total
    return lh, la


def implied_goals(df: pd.DataFrame) -> pd.DataFrame:
    """football-data rows -> columns hmkt, amkt (market-implied expected goals, NaN if no odds)."""
    p = market_probs(df)
    lh, la = implied_lambdas(p["ph"], p["pa"], p["pover"])
    return pd.DataFrame({"hmkt": lh, "amkt": la}, index=df.index)
