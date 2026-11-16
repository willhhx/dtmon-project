"""Publication figures 1-8. Each function writes a PNG (and PDF) into fig_dir and returns the path."""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from .models import get_model, KIND_NAMES
from .simulate import make_population, make_fault, simulate_truth, observe
from .filter import run_ekf
from .experiments import load_results, load_patients, SCHEME_ORDER
from .analysis import (add_patient_metrics, diagnosis_table, ablation_effects, variant_effects,
                       BASELINES_CONVENTIONAL, BASELINES_ADVANCED, CANDIDATES)
from .calibrate import roc_points

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 110})
KIND_COLORS = {"null": "#7f7f7f", "none": "#7f7f7f", "drift": "#4C72B0", "structure": "#DD8452", "feedback": "#55A868"}
DET_LABEL = {"D1_threshold": "D1 threshold", "D2_movavg": "D2 moving avg", "D3_ewma": "D3 EWMA",
             "D4_cusum": "D4 CUSUM", "D5_glr": "D5 GLR", "D6_sr": "D6 Shiryaev-Roberts",
             "D7_bocpd": "D7 BOCPD", "D8_conformal": "D8 conformal", "D9_multiscale": "D9 multi-scale",
             "D10_hybrid": "D10 hybrid"}
DET_COLOR = {d: c for d, c in zip(DET_LABEL, ["#9ecae1", "#6baed6", "#3182bd", "#08519c", "#fdae6b",
                                              "#fd8d3c", "#e6550d", "#a63603", "#31a354", "#006d2c"])}


def _save(fig, fig_dir, name):
    os.makedirs(fig_dir, exist_ok=True)
    p = os.path.join(fig_dir, name + ".png")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(fig_dir, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    return p


# ------------------------------------------------------------------------------ Figure 1
def fig1_pipeline(fig_dir):
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.axis("off")
    boxes = [(0.02, "Patient\n(sparse, irregular\nobservations y_t)", "#d9d9d9"),
             (0.21, "Digital twin\nODE + known inputs\n+ EKF observer", "#c6dbef"),
             (0.40, "Standardised\ninnovations z_t\n(per-patient normalised)", "#c7e9c0"),
             (0.59, "Sequential monitors\nD1-D10 (CUSUM, GLR, SR,\nBOCPD, conformal, multi-scale)", "#fdd0a2"),
             (0.80, "ALARM + fault isolation\ndrift | structure |\nfeedback loss", "#fcbba1")]
    for x, txt, col in boxes:
        ax.add_patch(FancyBboxPatch((x, 0.30), 0.16 if x < 0.79 else 0.18, 0.42, boxstyle="round,pad=0.01",
                                    fc=col, ec="k", lw=1, transform=ax.transAxes))
        ax.text(x + (0.08 if x < 0.79 else 0.09), 0.51, txt, ha="center", va="center", fontsize=8.5,
                transform=ax.transAxes)
    for x0, x1 in [(0.18, 0.21), (0.37, 0.40), (0.56, 0.59), (0.75, 0.80)]:
        ax.add_patch(FancyArrowPatch((x0, 0.51), (x1, 0.51), arrowstyle="-|>", mutation_scale=14,
                                     transform=ax.transAxes, color="k"))
    ax.text(0.5, 0.92, "Digital-twin monitoring pipeline", ha="center", fontsize=12, fontweight="bold",
            transform=ax.transAxes)
    ax.text(0.5, 0.12, "Thresholds calibrated on an independent no-failure cohort to a common false-alarm probability; "
            "diagnosis features: residual moments, autocorrelation, spectrum, state mismatch, uncertainty inflation.",
            ha="center", fontsize=8, transform=ax.transAxes, style="italic")
    return _save(fig, fig_dir, "fig1_pipeline")


# ------------------------------------------------------------------------------ Figure 2
def fig2_failure_examples(fig_dir, seed=3, noise=0.1):
    fig, axes = plt.subplots(2, 4, figsize=(15, 5.6))
    for r, name in enumerate(("glucose", "tumor")):
        m = get_model(name)
        pop = make_population(m, 12, seed)
        for c, (kind, label) in enumerate([(0, "No failure"), (1, "Parameter drift"),
                                           (2, "Structural error"), (3, "Feedback loss")]):
            ax = axes[r, c]
            base = simulate_truth(pop, make_fault(m, pop, 0))[0]
            f = make_fault(m, pop, kind, "large", sub=0 if kind == 2 else 0, seed=seed)
            y = simulate_truth(pop, f)[0]
            t = m.time_grid()
            sl = slice(int(0.35 * len(t)), None) if name == "glucose" else slice(0, None)
            if name == "glucose":      # zoom to 4 days around the change
                lo = int((m.t_change - 24) / m.dt)
                sl = slice(lo, lo + int(96 / m.dt))
            ax.plot(t[sl], base[sl], color="#7f7f7f", lw=1, label="no-failure twin trajectory")
            ax.plot(t[sl], y[sl], color=KIND_COLORS[KIND_NAMES[kind]], lw=1.3, label="patient (true)")
            co = observe(m, pop, simulate_truth(pop, f), f, "realistic" if name == "tumor" else "dense", noise, seed + 5)
            ob = co.y[0]
            idx = np.nonzero(np.isfinite(ob))[0]
            idx = idx[(idx >= sl.start if sl.start else True)]
            if name == "glucose":
                idx = idx[(idx >= sl.start) & (idx < sl.stop)][::3]
            ax.plot(t[idx], ob[idx], "k.", ms=3, label="observations")
            ax.axvline(f.onset[0] if kind else m.t_change, color="k", ls="--", lw=0.8)
            ax.set_title(f"{label}" + (f"  ({name})" if c == 0 else ""), fontsize=9)
            ax.set_xlabel("time (" + m.time_unit + ")")
            if c == 0:
                ax.set_ylabel("glucose (mg/dL)" if name == "glucose" else "tumour burden (a.u.)")
            if r == 0 and c == 0:
                ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("Failure injection examples (large magnitude; dashed = change time)", fontweight="bold")
    fig.tight_layout()
    return _save(fig, fig_dir, "fig2_failure_examples")


# ------------------------------------------------------------------------------ Figure 3
def residual_evolution(model_name, n=80, noise=0.1, scheme="dense", level="medium", seed=5, nbins=24):
    m = get_model(model_name)
    pop = make_population(m, n, seed)
    out = {}
    for kind in (0, 1, 2, 3):
        f = make_fault(m, pop, kind, level, seed=seed)
        co = observe(m, pop, simulate_truth(pop, f), f, scheme, noise, seed + 1)
        z = run_ekf(m, pop, co).z
        t = co.t
        onset = f.onset if kind else np.full(n, m.t_change)
        rel = t[None, :] - onset[:, None]
        edges = np.linspace(-0.3 * m.horizon, 0.45 * m.horizon, nbins + 1)
        mean, std, ctr = [], [], 0.5 * (edges[1:] + edges[:-1])
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = np.isfinite(z) & (rel >= lo) & (rel < hi) & (t[None, :] >= m.burn_in)
            v = z[sel]
            mean.append(v.mean() if v.size > 5 else np.nan)
            std.append(v.std() if v.size > 5 else np.nan)
        out[KIND_NAMES[kind]] = (ctr, np.array(mean), np.array(std))
    return out


def fig3_residual_evolution(fig_dir, **kw):
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex="col")
    for c, name in enumerate(("glucose", "tumor")):
        m = get_model(name)
        d = residual_evolution(name, **kw)
        for k, (ctr, mean, std) in d.items():
            axes[0, c].plot(ctr, mean, color=KIND_COLORS[k], marker="o", ms=3, label=k)
            axes[1, c].plot(ctr, std, color=KIND_COLORS[k], marker="o", ms=3, label=k)
        for r in range(2):
            axes[r, c].axvline(0, color="k", ls="--", lw=0.8)
        axes[0, c].set_title(f"{name}: mean of standardised innovation")
        axes[1, c].set_title(f"{name}: s.d. of standardised innovation")
        axes[1, c].set_xlabel(f"time since change ({m.time_unit})")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Residual evolution after failure (EKF innovations, medium magnitude)", fontweight="bold")
    fig.tight_layout()
    return _save(fig, fig_dir, "fig3_residual_evolution")


# ------------------------------------------------------------------------------ Figure 4
def fig4_detection_delay(results_dir, fig_dir, level="medium", detectors=None):
    P = add_patient_metrics(load_patients(results_dir))
    P = P[(P.level == level) & (P.variant == "full")]
    dets = [d for d in (detectors or list(DET_LABEL)) if d in set(P.detector)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 6.2))
    for r, model in enumerate(("glucose", "tumor")):
        unit = get_model(model).time_unit
        for c, kind in enumerate(("drift", "structure", "feedback")):
            ax = axes[r, c]
            g = P[(P.model == model) & (P.kind == kind)]
            if g.empty:
                ax.set_visible(False)
                continue
            cell = g.groupby(["detector", "cond_id"]).agg(delay=("delay_cens", "mean"), det=("det", "mean")).reset_index()
            agg = cell.groupby("detector").agg(delay=("delay", "mean"), sem=("delay", lambda x: x.std() / np.sqrt(max(len(x), 1))),
                                               det=("det", "mean")).reindex(dets)
            ax.bar(range(len(dets)), agg["delay"], yerr=agg["sem"], color=[DET_COLOR[d] for d in dets], capsize=2)
            for i, v in enumerate(agg["det"]):
                ax.text(i, 0, f"{v:.0%}", ha="center", va="bottom", fontsize=6, rotation=90, color="white")
            ax.set_xticks(range(len(dets)))
            ax.set_xticklabels([DET_LABEL[d].split(" ")[0] for d in dets], rotation=0, fontsize=7)
            ax.set_title(f"{model} / {kind}", fontsize=9)
            ax.set_ylabel(f"restricted mean delay ({unit})")
    fig.suptitle(f"Detection delay at equal false-alarm probability ({level} magnitude; misses censored at horizon; "
                 f"white labels = detection rate)", fontweight="bold", fontsize=10)
    fig.tight_layout()
    return _save(fig, fig_dir, "fig4_detection_delay")


# ------------------------------------------------------------------------------ Figure 5
def fig5_roc(results_dir, fig_dir, level="medium", detectors=("D4_cusum", "D5_glr", "D6_sr", "D7_bocpd",
                                                              "D8_conformal", "D9_multiscale", "D10_hybrid")):
    P = load_patients(results_dir)
    P = P[(P.level == level) & (P.variant == "full")]
    grid = np.linspace(0, 1, 101)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for r, model in enumerate(("glucose", "tumor")):
        for c, kind in enumerate(("drift", "structure", "feedback")):
            ax = axes[r, c]
            for det in detectors:
                g = P[(P.model == model) & (P.kind == kind) & (P.detector == det)]
                curves = []
                for cid, gg in g.groupby("cond_id"):
                    fpr, tpr = roc_points(gg.score_post.values, gg.score_null_post.values)
                    curves.append(np.interp(grid, fpr, tpr))
                if curves:
                    ax.plot(grid, np.mean(curves, axis=0), color=DET_COLOR[det], label=DET_LABEL[det], lw=1.4)
            ax.plot([0, 1], [0, 1], "k:", lw=0.8)
            ax.set_title(f"{model} / {kind}", fontsize=9)
            ax.set_xlabel("false positive rate")
            ax.set_ylabel("true positive rate")
    axes[0, 0].legend(fontsize=6.5, loc="lower right")
    fig.suptitle(f"ROC curves: max statistic after the change vs paired no-failure patients ({level}); averaged over conditions",
                 fontweight="bold", fontsize=10)
    fig.tight_layout()
    return _save(fig, fig_dir, "fig5_roc")


# ------------------------------------------------------------------------------ Figure 6
def fig6_confusion(results_dir, fig_dir, detector="D10_hybrid", classifier="F1_rf"):
    T = diagnosis_table(results_dir, detector, (classifier,))
    names = ["drift", "structure", "feedback"]
    models = [m for m in ("glucose", "tumor") if m in set(T.model)] if len(T) else []
    fig, axes = plt.subplots(1, max(len(models), 1), figsize=(5.2 * max(len(models), 1), 4.4))
    axes = np.atleast_1d(axes)
    if not models:
        axes[0].text(0.5, 0.5, "not enough detected failures\nfor diagnosis", ha="center", va="center")
    for ax, model in zip(axes, models):
        g = T[T.model == model]
        C = sum(np.array(json.loads(s)) for s in g.confusion)
        C = np.asarray(C, float)
        Cn = C / np.maximum(C.sum(1, keepdims=True), 1)
        ax.imshow(Cn, vmin=0, vmax=1, cmap="Blues")
        for i in range(Cn.shape[0]):
            for j in range(Cn.shape[1]):
                ax.text(j, i, f"{Cn[i, j]:.2f}\n(n={int(C[i, j])})", ha="center", va="center", fontsize=8,
                        color="white" if Cn[i, j] > 0.5 else "black")
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(names); ax.set_yticklabels(names)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(f"{model}: accuracy {np.trace(C) / max(C.sum(), 1):.2f}", fontsize=9)
        ax.grid(False)
    fig.suptitle(f"Fault-isolation confusion matrices ({detector} alarms, {classifier}; row-normalised)", fontweight="bold")
    fig.tight_layout()
    return _save(fig, fig_dir, "fig6_confusion")


# ------------------------------------------------------------------------------ Figure 7
def fig7_sparsity(results_dir, fig_dir, detectors=("D1_threshold", "D4_cusum", "D5_glr", "D6_sr", "D7_bocpd",
                                                   "D8_conformal", "D9_multiscale", "D10_hybrid")):
    S = load_results(results_dir)
    S = S[S.variant == "full"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for c, model in enumerate(("glucose", "tumor")):
        g = S[S.model == model]
        for r, (metric, lab) in enumerate((("detect_rate", "detection rate"), ("auroc", "AUROC"))):
            ax = axes[r, c]
            for det in detectors:
                d = g[g.detector == det]
                if d.empty:
                    continue
                a = d.groupby("scheme").agg(x=("mean_obs_post", "mean"), y=(metric, "mean")).sort_values("x")
                ax.plot(a.x, a.y, marker="o", ms=4, color=DET_COLOR.get(det), label=DET_LABEL.get(det, det))
            ax.set_xscale("log")
            ax.set_xlabel("mean number of observations after the change")
            ax.set_ylabel(lab)
            ax.set_title(f"{model}")
    axes[0, 0].legend(fontsize=6.5)
    fig.suptitle("Performance versus sampling sparsity (averaged over failure types, magnitudes and noise levels)",
                 fontweight="bold")
    fig.tight_layout()
    return _save(fig, fig_dir, "fig7_sparsity")


# ------------------------------------------------------------------------------ Figure 8
def fig8_ablation(results_dir, fig_dir):
    P = add_patient_metrics(load_patients(results_dir))
    A = ablation_effects(P)
    V = variant_effects(P)
    rows = []
    if len(A):
        for comp, g in A.groupby("component"):
            rows.append((comp, g.delta_detect.mean(), g.d_lo.mean(), g.d_hi.mean(), len(g)))
    if len(V):
        for comp, g in V.groupby("component"):
            rows.append((comp, g.effect.mean(), g.lo.mean(), g.hi.mean(), len(g)))
    fig, ax = plt.subplots(figsize=(8, 3.6 + 0.3 * len(rows)))
    if rows:
        lab = [r[0] for r in rows]
        eff = np.array([r[1] for r in rows])
        lo = np.array([r[2] for r in rows])
        hi = np.array([r[3] for r in rows])
        y = np.arange(len(rows))[::-1]
        ax.errorbar(eff, y, xerr=[eff - lo, hi - eff], fmt="o", color="#08519c", capsize=3)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{l} (n={r[4]})" for l, r in zip(lab, rows)])
        ax.set_xlabel("effect on detection rate of REMOVING the component (full - ablated), mean and mean 95% CI")
    else:
        ax.text(0.5, 0.5, "no ablation data in results directory", ha="center")
    ax.set_title("Ablation study", fontweight="bold")
    fig.tight_layout()
    return _save(fig, fig_dir, "fig8_ablation")


def make_all(results_dir, fig_dir):
    out = {}
    out["fig1"] = fig1_pipeline(fig_dir)
    out["fig2"] = fig2_failure_examples(fig_dir)
    out["fig3"] = fig3_residual_evolution(fig_dir)
    out["fig4"] = fig4_detection_delay(results_dir, fig_dir)
    out["fig5"] = fig5_roc(results_dir, fig_dir)
    out["fig6"] = fig6_confusion(results_dir, fig_dir)
    out["fig7"] = fig7_sparsity(results_dir, fig_dir)
    out["fig8"] = fig8_ablation(results_dir, fig_dir)
    return out
