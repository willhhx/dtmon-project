"""Figures for the paper-strengthening experiments (PNG + PDF). Matplotlib only."""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DET_SHOW = ["D4_cusum", "D6_sr", "D8_conformal", "D9_multiscale"]
DET_LABEL = {"D4_cusum": "CUSUM", "D6_sr": "Shiryaev-Roberts", "D8_conformal": "Conformal",
             "D9_multiscale": "Multi-scale", "D7_bocpd": "BOCPD", "D10_hybrid": "Hybrid"}
# colour-blind safe (Okabe-Ito) + distinct markers so the figure survives greyscale
COL = {"D4_cusum": "#0072B2", "D6_sr": "#E69F00", "D8_conformal": "#009E73", "D9_multiscale": "#CC79A7",
       "D7_bocpd": "#56B4E9", "D10_hybrid": "#D55E00"}
MARK = {"D4_cusum": "o", "D6_sr": "s", "D8_conformal": "^", "D9_multiscale": "D", "D7_bocpd": "v", "D10_hybrid": "P"}


def _save(fig, path_stem):
    os.makedirs(os.path.dirname(path_stem), exist_ok=True)
    fig.savefig(path_stem + ".png", dpi=200, bbox_inches="tight")
    fig.savefig(path_stem + ".pdf", bbox_inches="tight")
    plt.close(fig)
    return path_stem + ".png"


def fig_transfer(summary: pd.DataFrame, out_stem, detectors=DET_SHOW):
    """Top row: false-alarm rate of the nominal thresholds under each shift (with 95% CI, alpha dashed).
    Bottom row: power with the ORACLE threshold under the same shift (what the detector could do at alpha)."""
    models = list(summary.model.unique())
    shifts = list(dict.fromkeys(summary["shift"]))
    fig, axes = plt.subplots(2, len(models), figsize=(5.6 * len(models), 6.4), squeeze=False, sharex=True)
    x = np.arange(len(shifts))
    w = 0.8 / len(detectors)
    for c, m in enumerate(models):
        S = summary[summary.model == m]
        for i, d in enumerate(detectors):
            D = S[S.detector == d].set_index("shift").reindex(shifts)
            if D.far.isna().all():
                continue
            xs = x + (i - (len(detectors) - 1) / 2) * w
            axes[0, c].errorbar(xs, D.far, yerr=[D.far - D.far_lo, D.far_hi - D.far], fmt=MARK[d], color=COL[d],
                                ms=5, capsize=2, label=DET_LABEL.get(d, d))
            if "power_oracle_thr" in D:
                axes[1, c].bar(xs, D.power_oracle_thr, width=w * 0.9, color=COL[d], label=DET_LABEL.get(d, d))
        axes[0, c].axhline(S.alpha.iloc[0], color="k", ls="--", lw=1)
        axes[0, c].set_title(f"{m}: false alarms, nominal thresholds")
        axes[0, c].set_ylabel("false-alarm probability")
        axes[1, c].set_title(f"{m}: power at alpha (re-calibrated threshold)")
        axes[1, c].set_ylabel("detection probability")
        axes[1, c].set_xticks(x)
        axes[1, c].set_xticklabels(shifts, rotation=35, ha="right")
        axes[1, c].set_ylim(0, 1)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return _save(fig, out_stem)


def fig_burnin(df: pd.DataFrame, out_stem, detectors=("D4_cusum", "D6_sr", "D8_conformal", "D9_multiscale")):
    models = list(df.model.unique())
    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 3.8), squeeze=False)
    for c, m in enumerate(models):
        S = df[df.model == m]
        for d in detectors:
            D = S[S.detector == d].sort_values("burn_in_obs")
            if len(D):
                axes[0, c].plot(D.burn_in_obs, D.detect_rate, marker=MARK[d], color=COL[d], label=DET_LABEL.get(d, d))
        axes[0, c].set_xlabel("burn-in observations per patient")
        axes[0, c].set_ylabel("detection rate at alpha")
        axes[0, c].set_title(m)
        axes[0, c].set_ylim(0, 1)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return _save(fig, out_stem)


def fig_diag_baselines(summary: pd.DataFrame, out_stem, detector="D4_cusum"):
    S = summary[summary.detector == detector]
    models = list(S.model.unique())
    methods = list(dict.fromkeys(S.method))
    fig, axes = plt.subplots(1, len(models), figsize=(5.6 * len(models), 4.0), squeeze=False, sharey=True)
    for c, m in enumerate(models):
        D = S[S.model == m].groupby("method").agg(acc=("accuracy_given_detected", "mean"), lo=("acc_lo", "mean"),
                                                   hi=("acc_hi", "mean")).reindex(methods)
        y = np.arange(len(methods))
        axes[0, c].barh(y, D.acc, color="#0072B2")
        axes[0, c].errorbar(D.acc, y, xerr=[D.acc - D.lo, D.hi - D.acc], fmt="none", ecolor="k", capsize=2)
        axes[0, c].axvline(1 / 3, color="k", ls=":", lw=1)
        axes[0, c].set_yticks(y)
        axes[0, c].set_yticklabels(methods, fontsize=8)
        axes[0, c].set_xlim(0, 1)
        axes[0, c].set_xlabel("fault-isolation accuracy given detection")
        axes[0, c].set_title(m)
    fig.tight_layout()
    return _save(fig, out_stem)


def fig_realdata(far_df: pd.DataFrame, pw_df: pd.DataFrame, out_stem):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    F = far_df[far_df.threshold.str.startswith("calibrated")].set_index("detector")
    y = np.arange(len(F))
    axes[0].errorbar(F.far, y, xerr=[F.far - F.far_lo, F.far_hi - F.far], fmt="o", color="#0072B2", capsize=2)
    nom = far_df[~far_df.threshold.str.startswith("calibrated")]
    if len(nom):
        axes[0].plot(nom.far, [list(F.index).index("D6_sr")] * len(nom), "x", color="#D55E00", ms=9,
                     label="SR with nominal threshold (T1)")
        axes[0].legend(frameon=False, fontsize=8)
    axes[0].axvline(F.alpha.iloc[0], color="k", ls="--", lw=1)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(F.index, fontsize=8)
    axes[0].set_xlabel("false-alarm probability on held-out real patients")
    P = pw_df.groupby("detector").power.mean().reindex(F.index)
    axes[1].barh(y, P.values, color="#009E73")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(F.index, fontsize=8)
    axes[1].set_xlabel("mean power over injected perturbations")
    axes[1].set_xlim(0, 1)
    fig.tight_layout()
    return _save(fig, out_stem)
