"""Statistical analysis of experiment outputs: paired detector comparisons at equal false-alarm rate,
diagnosis experiments, ablation effect sizes, hypothesis scorecard and the GREEN/YELLOW/RED publication logic."""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd

from .models import get_model
from .experiments import load_results, load_patients
from .calibrate import paired_bootstrap, OUT_DET, OUT_FA
from .diagnose import (CLASSIFIERS, cross_validate_diagnosis, feature_names, GROUPS)

BASELINES_CONVENTIONAL = ["D1_threshold", "D2_movavg", "D3_ewma", "D4_cusum"]
BASELINES_ADVANCED = ["D5_glr", "D6_sr", "D7_bocpd", "D8_conformal"]
CANDIDATES = ["D9_multiscale", "D10_hybrid"]
KEYS = ["cond_id", "model", "noise", "scheme", "miss", "cal_sigma", "variant", "kind", "level"]


def horizons():
    return {n: get_model(n).horizon for n in ("glucose", "tumor")}


def add_patient_metrics(P: pd.DataFrame) -> pd.DataFrame:
    P = P.copy()
    hz = P["model"].map(horizons())
    P["det"] = (P["outcome"] == OUT_DET).astype(float)
    P["fa"] = (P["outcome"] == OUT_FA).astype(float)
    cens = hz - P["onset"]
    P["delay_cens"] = np.where(P["det"] == 1, P["delay"], cens)
    P.loc[P["fa"] == 1, "delay_cens"] = np.nan          # patients already alarmed before onset
    P["delay_rel"] = P["delay_cens"] / cens
    return P


def summary_table(S: pd.DataFrame, by=("model", "scheme", "noise", "kind", "level", "detector"),
                  metrics=("detect_rate", "median_delay", "mean_delay_cens", "auroc", "far_null_full")):
    return S.groupby(list(by))[list(metrics)].mean().reset_index()


# ---------------------------------------------------------------------------------------------
def compare_detectors(P: pd.DataFrame, cand: str, base: str, n_boot=1000, seed=0,
                      group_cols=("cond_id", "kind", "level")) -> pd.DataFrame:
    """Paired (same-patient) comparison of two detectors in every condition cell.
    delta_detect = P(detect | cand) - P(detect | base); delta_delay_rel = relative restricted delay diff."""
    rows = []
    a = P[P.detector == cand]
    b = P[P.detector == base]
    for key, ga in a.groupby(list(group_cols)):
        gb = b
        for c, v in zip(group_cols, key):
            gb = gb[gb[c] == v]
        if gb.empty:
            continue
        m = ga.merge(gb, on="patient", suffixes=("_a", "_b"))
        if m.empty:
            continue
        d, lo, hi = paired_bootstrap(m["det_a"].values, m["det_b"].values, n_boot, seed)
        dd, dlo, dhi = paired_bootstrap(m["delay_rel_a"].values, m["delay_rel_b"].values, n_boot, seed)
        r = dict(zip(group_cols, key))
        r.update(model=ga["model"].iloc[0], scheme=ga["scheme"].iloc[0], noise=ga["noise"].iloc[0],
                 variant=ga["variant"].iloc[0], cand=cand, base=base, n=len(m),
                 det_cand=m["det_a"].mean(), det_base=m["det_b"].mean(),
                 delta_detect=d, d_lo=lo, d_hi=hi, delta_delay_rel=dd, dd_lo=dlo, dd_hi=dhi)
        rows.append(r)
    return pd.DataFrame(rows)


def best_baseline_per_cell(P: pd.DataFrame, baselines) -> pd.DataFrame:
    """Name of the baseline with the highest detection rate in each (cond, kind, level) cell."""
    g = P[P.detector.isin(baselines)].groupby(["cond_id", "kind", "level", "detector"])["det"].mean().reset_index()
    idx = g.groupby(["cond_id", "kind", "level"])["det"].idxmax()
    return g.loc[idx, ["cond_id", "kind", "level", "detector"]].rename(columns={"detector": "best_base"})


def compare_to_best(P: pd.DataFrame, cand: str, baselines, n_boot=1000, seed=0, select="model") -> pd.DataFrame:
    """Compare a candidate against the strongest baseline.
    select='model': ONE baseline per model, chosen by its mean detection rate over all cells (pre-specified,
                    unbiased benchmark; default).
    select='cell' : the best baseline separately in every cell (oracle per cell; optimistic for the baselines,
                    use only as a conservative sensitivity analysis)."""
    rows = []
    if select == "cell":
        bb = best_baseline_per_cell(P, baselines)
        for _, r in bb.iterrows():
            sub = P[(P.cond_id == r.cond_id) & (P.kind == r.kind) & (P.level == r.level)]
            rows.append(compare_detectors(sub, cand, r.best_base, n_boot, seed))
    else:
        for model, g in P.groupby("model"):
            means = g[g.detector.isin(baselines)].groupby("detector")["det"].mean()
            if means.empty:
                continue
            best = means.idxmax()
            c = compare_detectors(g[g.detector.isin([cand, best])], cand, best, n_boot, seed)
            c["best_base_model"] = best
            rows.append(c)
    rows = [r for r in rows if len(r)]
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------------------------
def classify_outcome(delta: float, lo: float, hi: float, min_effect=0.05) -> str:
    """GREEN: effect >= min_effect with CI excluding 0 | YELLOW: positive point estimate | RED: none."""
    if np.isfinite(lo) and lo > 0 and delta >= min_effect:
        return "GREEN"
    if delta > 0.01:
        return "YELLOW"
    return "RED"


RED_PROPOSALS = [
    "Stronger baselines: add a likelihood-ratio detector with the TRUE post-change model (oracle upper bound) and a "
    "Kalman-residual chi-square (windowed NIS) test; confirm the candidate is not just matching them.",
    "Alternative detectors: add a kernel / MMD sequential test, an e-process (mixture martingale) detector, "
    "and a learned (neural/ensemble) detector trained on simulated failures.",
    "Revised hypothesis: restrict the claim to the regime where the candidate wins (e.g. periodic structural errors, "
    "or sparse sampling) and demonstrate it over a finer sparsity grid.",
    "Increase failure realism / difficulty: smaller magnitudes, mixed simultaneous failures, longer ramps; "
    "advantages of multi-scale methods usually appear for weak, slowly-developing changes.",
    "Strengthen the diagnostic contribution: report fault-isolation accuracy at EQUAL detection delay, "
    "which is often where residual-signature methods separate from CUSUM.",
]


def publication_decision(P: pd.DataFrame, candidates=CANDIDATES, baselines=None, n_boot=500, seed=0,
                         min_effect=0.05, sparse_schemes=("sparse", "realistic")) -> dict:
    """After an experiment, label GREEN/YELLOW/RED for success criteria A (detection power at equal FAR /
    delay) and C (robustness to sparse sampling), per model, and criterion D (transferability)."""
    baselines = baselines or (BASELINES_CONVENTIONAL + BASELINES_ADVANCED)
    out = {"details": []}
    for cand in candidates:
        cmp_ = compare_to_best(P, cand, baselines, n_boot=n_boot, seed=seed, select="model")
        if cmp_.empty:
            continue
        sens = compare_to_best(P, cand, baselines, n_boot=max(50, n_boot // 5), seed=seed, select="cell")
        cmp_["label"] = [classify_outcome(d, lo, hi, min_effect) for d, lo, hi in
                         zip(cmp_.delta_detect, cmp_.d_lo, cmp_.d_hi)]
        per_model = {}
        for model, g in cmp_.groupby("model"):
            A_green = (g.label == "GREEN").mean()
            gs = g[g.scheme.isin(sparse_schemes)]
            C_green = (gs.label == "GREEN").mean() if len(gs) else np.nan
            delay_gain = -g.delta_delay_rel.mean()
            per_model[model] = dict(
                A_frac_green=float(A_green), mean_delta_detect=float(g.delta_detect.mean()),
                mean_delay_reduction=float(delay_gain), C_frac_green_sparse=float(C_green),
                oracle_cell_mean_delta=float(sens[sens.model == model].delta_detect.mean()) if len(sens) else np.nan,
                best_baseline=str(g.best_base_model.iloc[0]), n_cells=int(len(g)))
        met = {m: (v["A_frac_green"] >= 0.25) or (np.nan_to_num(v["C_frac_green_sparse"]) >= 0.25)
               for m, v in per_model.items()}
        pos = {m: v["mean_delta_detect"] > 0.01 for m, v in per_model.items()}
        if len(per_model) >= 2 and all(met.values()):
            label = "GREEN"
        elif any(met.values()):
            label = "YELLOW"
        elif any(pos.values()):
            label = "YELLOW"
        else:
            label = "RED"
        out["details"].append(dict(candidate=cand, label=label, per_model=per_model,
                                   criterion_D_transferable=bool(len(per_model) >= 2 and all(met.values()))))
        out[cand] = label
        if label == "RED":
            out.setdefault("proposals", RED_PROPOSALS)
    return out


# ---------------------------------------------------------------------------------------------
def load_diag(out_dir: str):
    """Load diagnosis feature files together with the condition metadata."""
    S = load_results(out_dir)
    meta = S.drop_duplicates("cond_id").set_index("cond_id")[["model", "noise", "scheme", "variant", "n_patients"]]
    d = {}
    for f in sorted(os.listdir(out_dir)):
        if f.startswith("diag_") and f.endswith(".npz"):
            cid = f[5:-4]
            if cid in meta.index:
                d[cid] = (dict(np.load(os.path.join(out_dir, f), allow_pickle=True)), meta.loc[cid].to_dict())
    return d


def diagnosis_table(out_dir: str, detector="D10_hybrid", classifiers=CLASSIFIERS, groups=GROUPS,
                    pool_by=("model", "scheme"), seed=0, use_sequences=True) -> pd.DataFrame:
    """Fault-isolation accuracy (3 classes) from residual features at the detector's alarm.
    Cross-validation is grouped by (condition, patient). Reports detection-conditional accuracy and
    end-to-end accuracy (detected AND correctly isolated / all failed patients)."""
    D = load_diag(out_dir)
    names = feature_names(GROUPS)
    sel = [i for i, n in enumerate(names) if n in feature_names(groups)]
    pools = {}
    for cid, (arr, meta) in D.items():
        if f"{detector}__X" not in arr:
            continue
        key = tuple(meta[k] for k in pool_by)
        pools.setdefault(key, []).append((cid, arr, meta))
    rows = []
    for key, items in pools.items():
        X = np.concatenate([a[f"{detector}__X"] for _, a, _ in items])
        S = np.concatenate([a[f"{detector}__seq"] for _, a, _ in items])
        y = np.concatenate([a[f"{detector}__kind"] for _, a, _ in items]).astype(int)
        grp = np.concatenate([[f"{cid}_{p}" for p in a[f"{detector}__pid"]] for cid, a, _ in items])
        n_total = sum(int(m["n_patients"]) * 9 for _, _, m in items)         # 3 kinds x 3 levels
        if len(y) < 30 or len(np.unique(y)) < 2:
            continue
        for clf in classifiers:
            Xin = S if (clf == "F4_neural" and use_sequences) else X[:, sel]
            res = cross_validate_diagnosis(Xin, y, grp, clf, seed=seed)
            if res is None:
                continue
            rows.append(dict(zip(pool_by, key), detector=detector, classifier=clf, groups="+".join(groups),
                             n_detected=len(y), n_failed_total=n_total,
                             accuracy_given_detected=res["accuracy"], macro_f1=res["macro_f1"],
                             end_to_end_accuracy=res["accuracy"] * len(y) / n_total, ece=res["ece"],
                             backend=res["backend"], confusion=json.dumps(res["confusion"].tolist())))
    return pd.DataFrame(rows)


def diagnosis_ablation(out_dir, detector="D10_hybrid", clf="F1_rf", seed=0):
    """Remove one feature group at a time and report the accuracy change (effect size)."""
    base = diagnosis_table(out_dir, detector, (clf,), GROUPS, seed=seed)
    rows = []
    for g in GROUPS:
        keep = tuple(x for x in GROUPS if x != g)
        t = diagnosis_table(out_dir, detector, (clf,), keep, seed=seed)
        m = base.merge(t, on=["model", "scheme"], suffixes=("", "_drop"))
        for _, r in m.iterrows():
            rows.append(dict(model=r["model"], scheme=r["scheme"], removed=g,
                             acc_full=r["accuracy_given_detected"], acc_without=r["accuracy_given_detected_drop"],
                             effect=r["accuracy_given_detected"] - r["accuracy_given_detected_drop"]))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
def ablation_effects(P: pd.DataFrame) -> pd.DataFrame:
    """Detector-level ablations: effect (full - ablated) on detection rate, with paired bootstrap CIs."""
    pairs = [("D9_multiscale", "D9_multiscale[-frequency]", "frequency features"),
             ("D9_multiscale", "D9_multiscale[single-scale]", "multi-scale residuals"),
             ("D10_hybrid", "D10_hybrid[-conformal]", "conformal calibration"),
             ("D10_hybrid", "D10_hybrid[-multiscale]", "multi-scale member")]
    rows = []
    for full, abl, label in pairs:
        if abl not in set(P.detector):
            continue
        c = compare_detectors(P[P.detector.isin([full, abl])], full, abl)
        if len(c):
            c["component"] = label
            rows.append(c)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def variant_effects(P: pd.DataFrame, detector="D10_hybrid") -> pd.DataFrame:
    """Effect of 'uncertainty normalization' (full vs no_norm) and 'nonlinear correction' (EKF vs open-loop)."""
    rows = []
    key = ["model", "noise", "scheme", "kind", "level", "patient"]
    d = P[P.detector == detector]
    full = d[d.variant == "full"][key + ["det"]]
    for var, label in (("no_norm", "uncertainty normalization"), ("open_loop", "nonlinear (EKF) correction")):
        o = d[d.variant == var][key + ["det"]]
        if o.empty:
            continue
        m = full.merge(o, on=key, suffixes=("_full", "_abl"))
        for k, g in m.groupby(["model", "scheme", "noise", "kind", "level"]):
            dlt, lo, hi = paired_bootstrap(g["det_full"].values, g["det_abl"].values, 500, 0)
            rows.append(dict(zip(["model", "scheme", "noise", "kind", "level"], k), component=label,
                             effect=dlt, lo=lo, hi=hi, n=len(g)))
    return pd.DataFrame(rows)


def hypothesis_scorecard(out_dir: str, min_effect=0.05) -> pd.DataFrame:
    """Evaluate H1-H5 from a results directory (see docs/hypotheses.md for the definitions)."""
    P = add_patient_metrics(load_patients(out_dir))
    S = load_results(out_dir)
    rows = []
    # H1: multi-scale detects structural failures missed by standard CUSUM
    Ps = P[P.kind == "structure"]
    c = compare_detectors(Ps, "D9_multiscale", "D4_cusum")
    rows.append(_hyp_row("H1 multi-scale > CUSUM on structural failures", c, min_effect))
    # H4: BOCPD improves sparse-sampling performance
    Psp = P[P.scheme.isin(["sparse", "realistic"])]
    c = compare_detectors(Psp, "D7_bocpd", "D4_cusum")
    rows.append(_hyp_row("H4 BOCPD > CUSUM under sparse sampling", c, min_effect))
    # H3: conformal calibration controls false alarms (within-condition calibration error)
    d8 = S[S.detector == "D8_conformal"]
    err8 = (d8.far_null_full - d8.alpha).abs().mean() if len(d8) else np.nan
    oth = S[~S.detector.isin(["D8_conformal"])]
    erro = (oth.far_null_full - oth.alpha).abs().mean() if len(oth) else np.nan
    rows.append(dict(hypothesis="H3 conformal alarm keeps FA at nominal (|FA-alpha|, lower=better)",
                     effect=float(erro - err8) if np.isfinite(err8) else np.nan, ci_lo=np.nan, ci_hi=np.nan,
                     label="YELLOW" if np.isfinite(err8) and err8 <= erro else "RED",
                     note=f"D8 {err8:.3f} vs others {erro:.3f}; use `dtmon transfer` for the misspecification test"))
    # H2/H5 from diagnosis tables
    try:
        full = diagnosis_table(out_dir, "D10_hybrid", ("F1_rf",), GROUPS)
        nofreq = diagnosis_table(out_dir, "D10_hybrid", ("F1_rf",), tuple(g for g in GROUPS if g != "spectrum"))
        if len(full) and len(nofreq):
            m = full.merge(nofreq, on=["model", "scheme"], suffixes=("", "_nf"))
            eff = float((m.accuracy_given_detected - m.accuracy_given_detected_nf).mean())
            rows.append(dict(hypothesis="H2 frequency features improve diagnosis (acc. gain)", effect=eff,
                             ci_lo=np.nan, ci_hi=np.nan, label=classify_outcome(eff, eff if eff > 0 else -1, np.nan, min_effect),
                             note="gain in detection-conditional accuracy, F1_rf"))
        shared = diagnosis_table(out_dir, "D4_cusum", ("F1_rf",), GROUPS)
        if len(shared):
            acc = float(shared.accuracy_given_detected.mean())
            rows.append(dict(hypothesis="H5 fault signatures isolate faults with a SHARED detection statistic (CUSUM alarms)",
                             effect=acc - 1 / 3, ci_lo=np.nan, ci_hi=np.nan,
                             label="GREEN" if acc >= 0.8 else ("YELLOW" if acc >= 0.55 else "RED"),
                             note=f"accuracy {acc:.2f} vs chance 0.33"))
    except Exception as e:  # pragma: no cover
        rows.append(dict(hypothesis="H2/H5", effect=np.nan, ci_lo=np.nan, ci_hi=np.nan, label="N/A", note=str(e)))
    return pd.DataFrame(rows)


def _hyp_row(name, c, min_effect):
    if c.empty:
        return dict(hypothesis=name, effect=np.nan, ci_lo=np.nan, ci_hi=np.nan, label="N/A", note="no data")
    d = float(c.delta_detect.mean())
    lo = float(c.d_lo.mean())
    hi = float(c.d_hi.mean())
    frac = float((c.d_lo > 0).mean())
    return dict(hypothesis=name, effect=d, ci_lo=lo, ci_hi=hi,
                label=classify_outcome(d, lo, hi, min_effect),
                note=f"{frac:.0%} of {len(c)} cells have CI > 0; mean delay change {c.delta_delay_rel.mean():+.2f} (rel.)")
