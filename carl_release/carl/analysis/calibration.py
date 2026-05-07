"""V(s_0) calibration metrics (paper Appendix H).

Computes per-decile reliability bins, ECE, Brier score, and AUC for separating
Tier-1 from Tier-2 questions.
"""
from __future__ import annotations


def compute_calibration(v_scores: list[float], outcomes: list[int],
                        n_bins: int = 10) -> list[dict]:
    """Returns one row per bin: {lo, hi, n, mean_pred, mean_actual}."""
    rows = []
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        bin_preds, bin_actual = [], []
        for v, y in zip(v_scores, outcomes):
            if (lo <= v < hi) or (i == n_bins - 1 and v == 1.0):
                bin_preds.append(v); bin_actual.append(y)
        if bin_preds:
            rows.append({"lo": lo, "hi": hi, "n": len(bin_preds),
                         "mean_pred": sum(bin_preds) / len(bin_preds),
                         "mean_actual": sum(bin_actual) / len(bin_actual)})
        else:
            rows.append({"lo": lo, "hi": hi, "n": 0, "mean_pred": 0.0, "mean_actual": 0.0})
    return rows


def ece(v_scores, outcomes, n_bins: int = 10) -> float:
    rows = compute_calibration(v_scores, outcomes, n_bins)
    n = sum(r["n"] for r in rows)
    if n == 0: return 0.0
    return sum(r["n"] / n * abs(r["mean_pred"] - r["mean_actual"]) for r in rows)


def brier(v_scores, outcomes) -> float:
    if not v_scores: return 0.0
    return sum((v - y) ** 2 for v, y in zip(v_scores, outcomes)) / len(v_scores)


def auc_t1_vs_t2(v_scores, tiers) -> float:
    """tiers: 'tier1' or 'tier2'. AUC of v_scores discriminating tier2 (positive) vs tier1."""
    pos = [v for v, t in zip(v_scores, tiers) if t == "tier2"]
    neg = [v for v, t in zip(v_scores, tiers) if t == "tier1"]
    if not pos or not neg: return 0.5
    n = 0
    for p in pos:
        for ng in neg:
            n += int(p > ng) + 0.5 * int(p == ng)
    return n / (len(pos) * len(neg))
