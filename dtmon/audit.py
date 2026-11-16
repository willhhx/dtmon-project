"""Sanity gate: catch degenerate detectors BEFORE any result is interpreted.

Two bugs in this project looked like findings: BOCPD (D7) had a false-alarm rate of exactly 0 in every cell
because its statistic sat at a ceiling, and conformal (D8) had a false-alarm rate of exactly 0 in glucose
because its p-values were tied. A detector that never alarms looks perfectly 'robust'. The audit flags

  DEAD_FAR     false-alarm rate well below alpha although thresholds were calibrated to alpha (ties / ceiling)
  LIBERAL      false-alarm rate well above alpha (calibration failed)
  NO_POWER     detection rate ~0 even for LARGE failures (the detector never fires)

Usage:  python -m dtmon audit --results results/main_P500 [--transfer results/transfer2]
A non-empty flag table means: do not interpret that detector's rows (fix or drop it first).
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _se(alpha, n):
    """Standard error of an empirical false-alarm rate when the threshold was ALSO calibrated on n patients:
    variance = alpha(1-alpha)(1/n_eval + 1/n_cal)."""
    return np.sqrt(2.0 * alpha * (1 - alpha) / np.maximum(n, 1))


def audit_results(S: pd.DataFrame, alpha: float = 0.05, k_sigma: float = 3.0, min_power: float = 0.03) -> pd.DataFrame:
    """Audit the per-condition summaries written by `main`/`ablation`/`stress`.
    Returns one row per (model, scheme, detector) with flags."""
    if S.empty:
        return pd.DataFrame()
    S = S[S.variant == "full"] if "variant" in S else S
    far = (S.drop_duplicates(["cond_id", "detector"])
             [["cond_id", "model", "scheme", "noise", "detector", "far_null_full", "n_patients"]])
    g = far.groupby(["model", "scheme", "detector"])
    A = g.agg(n_cond=("cond_id", "nunique"), mean_far=("far_null_full", "mean"),
              min_far=("far_null_full", "min"), max_far=("far_null_full", "max"),
              n_patients=("n_patients", "min")).reset_index()
    big = S[S.level == "large"] if "level" in S else S
    pw = (big.groupby(["model", "scheme", "detector"])["detect_rate_nofa"].mean().rename("mean_detect_large").reset_index())
    A = A.merge(pw, on=["model", "scheme", "detector"], how="left")
    flags = []
    for _, r in A.iterrows():
        f = []
        s = _se(alpha, r.n_patients)
        if (r.mean_far < alpha - k_sigma * s) or (r.min_far == 0.0 and r.n_patients * alpha >= 10):
            f.append("DEAD_FAR" if r.mean_far < alpha / 4 else "CONSERVATIVE")
        if r.mean_far > alpha + k_sigma * s:
            f.append("LIBERAL")
        if np.isfinite(r.mean_detect_large) and r.mean_detect_large < min_power:
            f.append("NO_POWER")
        flags.append(",".join(f))
    A["flag"] = flags
    A["alpha"] = alpha
    return A


def audit_transfer(far_df: pd.DataFrame, alpha: float = 0.05, k_sigma: float = 3.0) -> pd.DataFrame:
    """Audit the nominal-condition rows of a transfer table (`far` per detector). Needs columns
    model, shift, detector, detected_fa/n (transfer2) or far (legacy)."""
    d = far_df[far_df["shift"] == "nominal"].copy() if "shift" in far_df else far_df.copy()
    if "n" not in d:
        d["n"] = 200
    d["far"] = d["far"].astype(float)
    d["flag"] = np.where(d["far"] < alpha - k_sigma * _se(alpha, d["n"]), "CONSERVATIVE/DEAD",
                         np.where(d["far"] > alpha + k_sigma * _se(alpha, d["n"]), "LIBERAL", ""))
    return d


def tie_fraction(stat: np.ndarray, Tm: np.ndarray, h: float) -> float:
    """Fraction of no-failure patients whose maximum statistic EQUALS the threshold (ties at the threshold make
    the achieved false-alarm probability fall below alpha)."""
    s = np.where(np.isnan(stat), -np.inf, stat)
    mx = s.max(axis=1)
    mx = mx[np.isfinite(mx)]
    return float(np.mean(mx == h)) if mx.size else float("nan")


def render(A: pd.DataFrame) -> str:
    if A.empty:
        return "audit: no data"
    bad = A[A.flag != ""]
    lines = [f"audit: {len(A)} (model, scheme, detector) groups checked, {len(bad)} flagged"]
    if len(bad):
        lines.append(bad[["model", "scheme", "detector", "mean_far", "mean_detect_large", "flag"]]
                     .round(3).to_string(index=False))
        lines.append("AUDIT FAILED: do not interpret flagged detectors until fixed.")
    else:
        lines.append("AUDIT PASSED")
    return "\n".join(lines)
