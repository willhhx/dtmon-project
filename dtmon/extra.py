"""Paper-strengthening experiments (what a reviewer will ask for).

1. power_transfer   H3 done properly: calibrate at NOMINAL conditions, then under each shifted condition measure
                    (a) false alarms of the nominal thresholds and (b) POWER, both with the nominal threshold and
                    with an oracle threshold re-calibrated on the shifted condition. A detector that merely never
                    alarms cannot hide: its power is reported next to its false-alarm rate.
2. burnin_curve     price of conformal (within-patient) calibration: power vs number of burn-in observations.
3. diagnosis_baselines   fault isolation vs simple baselines (majority class, depth-2 rule, LDA, RF on moments only,
                    RF on all features), end-to-end accuracy, per-class / per-level / per-structural-subtype recall,
                    patient-level bootstrap CIs.
All functions return DataFrames; the CLI (`python -m dtmon transfer2|burnin|diag2`) writes them to csv.
"""
from __future__ import annotations
import contextlib
import numpy as np
import pandas as pd

from .models import get_model, MODELS, KIND_NAMES, SUB_NAMES
from .simulate import make_population, observe
from .experiments import _truth, _residuals, run_condition
from .detectors import all_detectors, CUSUM, GLR, ShiryaevRoberts, ConformalAlarm, MultiScale, BOCPD
from .calibrate import calibrate_threshold, evaluate, per_patient_max, OUT_DET
from .sampling import SCHEMES

NOMINAL = dict(noise=0.1, scheme="moderate", cal_sigma=0.05, cv_assumed_scale=1.0, miss=0.0)


def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(max(c - h, 0.0)), float(min(c + h, 1.0))


def default_shifts(nominal=None):
    nominal = dict(NOMINAL if nominal is None else nominal)
    return [("nominal", dict(nominal)),
            ("noise x2", dict(nominal, noise=0.2)),
            ("noise x4", dict(nominal, noise=0.4)),
            ("noise model x0.5", dict(nominal, cv_assumed_scale=0.5)),
            ("noise model x2", dict(nominal, cv_assumed_scale=2.0)),
            ("param error 15%", dict(nominal, cal_sigma=0.15)),
            ("missing 30%", dict(nominal, miss=0.3)),
            ("irregular visits", dict(nominal, scheme="realistic"))]


# ----------------------------------------------------------------------------------------------
def _pop(model, tc, P, pop_seed):
    return make_population(model, P, pop_seed, tc.get("cv_scale", 1.0), tc.get("cal_sigma", 0.05),
                           tc.get("p0_frac", 0.05), tc.get("treat_frac", 0.5))


def _res(model, pop, co, tc):
    return _residuals(model, pop, co, tc.get("mode", "ekf"), tc.get("normalize", True), tc.get("q_frac", 0.03),
                      tc.get("p0_frac", 0.05), tc.get("cv_assumed_scale", 1.0))


def _obs(model, pop, y, f, tc, seed):
    return observe(model, pop, y, f, tc.get("scheme", "dense"), tc.get("noise", 0.1), seed, tc.get("miss", 0.0),
                   tc.get("miss_kind", "random"), tc.get("freq_cv", 0.0))


def _null_R(model, tc, P, pop_seed, obs_seed, cache_dir):
    pop = _pop(model, tc, P, pop_seed)
    y, f = _truth(model, pop, 0, "medium", 0, pop_seed, P, cache_dir)
    return _res(model, pop, _obs(model, pop, y, f, tc, obs_seed), tc)


def _shift_worker(j, label, tc, model_name, dets, thr, P, seed, alpha, kinds, levels, cache_dir):
    model = get_model(model_name)
    pop_seed = seed + 5000 + j
    far_rows, power_rows = [], []
    # (a) false alarms of the NOMINAL thresholds under the shifted condition
    R0 = _null_R(model, tc, P, pop_seed, seed + 100 + j, cache_dir)
    for d in dets:
        mx = per_patient_max(d.stat_cached(R0), R0.Tm)
        k = int(np.sum(mx > thr[d.name]))
        lo, hi = wilson(k, P)
        far_rows.append(dict(model=model_name, shift=label, detector=d.name, false_alarms=k, n=P,
                             far=k / P, far_lo=lo, far_hi=hi, alpha=alpha))
    del R0
    # (b) oracle thresholds: calibrated on an independent no-failure cohort under the SHIFTED condition
    R1 = _null_R(model, tc, P, seed + 7000 + j, seed + 300 + j, cache_dir)
    thr_or = {d.name: calibrate_threshold(d.stat_cached(R1), R1.Tm, alpha) for d in dets}
    del R1
    # (c) power on failure cohorts under the shifted condition
    pop = _pop(model, tc, P, pop_seed)
    for kind in kinds:
        for lv in levels:
            y, f = _truth(model, pop, kind, lv, "random", pop_seed, P, cache_dir)
            R = _res(model, pop, _obs(model, pop, y, f, tc, seed + 400 + j), tc)
            for d in dets:
                s = d.stat_cached(R)
                for tag, th in (("nominal_thr", thr[d.name]), ("oracle_thr", thr_or[d.name])):
                    ev = evaluate(s, R.Tm, th, f.onset, model.horizon)
                    det = ev["outcome"] == OUT_DET
                    power_rows.append(dict(model=model_name, shift=label, kind=KIND_NAMES[kind], level=lv,
                                           detector=d.name, threshold=tag, detected=int(det.sum()), n=P,
                                           power=float(det.mean()),
                                           median_delay=float(np.nanmedian(ev["delay"])) if det.any() else np.nan))
            del R
    return far_rows, power_rows


def power_transfer(model_name, shifts=None, n_patients=200, seed=0, alpha=0.05, detectors=None,
                   kinds=(1, 2, 3), levels=("medium",), cache_dir=None, n_jobs=1, nominal=None):
    """H3 with power. Returns (far_df, power_df, summary_df)."""
    nominal = dict(NOMINAL if nominal is None else nominal)
    shifts = shifts if shifts is not None else default_shifts(nominal)
    model = get_model(model_name)
    dets = detectors if detectors is not None else all_detectors()
    P = n_patients
    R = _null_R(model, nominal, P, seed + 1000, seed + 1, cache_dir)
    thr = {}
    for d in dets:
        d.fit(R)
        thr[d.name] = calibrate_threshold(d.stat_cached(R), R.Tm, alpha)
    del R
    args = [(j, lab, tc) for j, (lab, tc) in enumerate(shifts)]
    if n_jobs == 1:
        out = [_shift_worker(j, lab, tc, model_name, dets, thr, P, seed, alpha, kinds, levels, cache_dir)
               for j, lab, tc in args]
    else:
        from joblib import Parallel, delayed
        out = Parallel(n_jobs=n_jobs)(delayed(_shift_worker)(j, lab, tc, model_name, dets, thr, P, seed, alpha,
                                                             kinds, levels, cache_dir) for j, lab, tc in args)
    far_df = pd.DataFrame([r for o in out for r in o[0]])
    power_df = pd.DataFrame([r for o in out for r in o[1]])
    return far_df, power_df, summarize_transfer(far_df, power_df)


def summarize_transfer(far_df: pd.DataFrame, power_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, shift, detector): false-alarm rate with CI, and mean power over failure types with the
    nominal threshold and with the oracle (re-calibrated) threshold."""
    g = (power_df.groupby(["model", "shift", "detector", "threshold"])
         .agg(detected=("detected", "sum"), n=("n", "sum")).reset_index())
    g["power"] = g.detected / g.n
    ci = g.apply(lambda r: wilson(r.detected, r.n), axis=1, result_type="expand")
    g["power_lo"], g["power_hi"] = ci[0], ci[1]
    w = g.pivot_table(index=["model", "shift", "detector"], columns="threshold",
                      values=["power", "power_lo", "power_hi"]).reset_index()
    w.columns = ["_".join([c for c in col if c]) for col in w.columns.values]
    out = far_df.merge(w, on=["model", "shift", "detector"], how="left")
    # a detector is "safe" if its false-alarm CI contains alpha or lies below it, and "useful" if the oracle
    # power is not negligible; "dead" detectors are safe but not useful.
    out["safe"] = out.far_lo <= out.alpha * 1.5
    out["useful"] = out.get("power_oracle_thr", np.nan) >= 0.2
    return out


# ----------------------------------------------------------------------------------------------
@contextlib.contextmanager
def burn_in_override(model_name, value):
    """Temporarily change the burn-in length (model time units) of a model class."""
    cls = MODELS[model_name]
    old = cls.burn_in
    cls.burn_in = float(value)
    try:
        yield
    finally:
        cls.burn_in = old


def burnin_curve(model_name, factors=(0.5, 1.0, 1.5, 2.0), scheme="moderate", noise=0.1, n_patients=200, seed=0,
                 alpha=0.05, levels=("medium", "large"), cache_dir=None):
    """Detection rate (at equal false-alarm probability) vs length of the within-patient burn-in. Conformal needs
    burn-in scores to resolve small p-values and the per-patient normalisation needs them too."""
    base = MODELS[model_name].burn_in
    interval = SCHEMES[model_name][scheme]
    rows = []
    for f in factors:
        with burn_in_override(model_name, base * f):
            dets = [CUSUM(), ShiryaevRoberts(), ConformalAlarm(), MultiScale(), BOCPD()]
            res = run_condition(dict(model=model_name, noise=noise, scheme=scheme), n_patients=n_patients, seed=seed,
                                alpha=alpha, levels=levels, detectors=dets, cache_dir=cache_dir, store_diag=False,
                                verbose=False)
        S = res["summary"]
        g = S.groupby("detector").agg(detect_rate=("detect_rate_nofa", "mean"), far_null=("far_null_full", "mean"),
                                      auroc=("auroc", "mean")).reset_index()
        g.insert(0, "model", model_name)
        g.insert(1, "burn_in", base * f)
        g.insert(2, "burn_in_obs", base * f / interval)
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


# ----------------------------------------------------------------------------------------------
def _acc_ci(correct, gid, B=1000, seed=0):
    """Accuracy with a patient-level (cluster) bootstrap CI."""
    correct = np.asarray(correct, float)
    if correct.size == 0:
        return float("nan"), float("nan"), float("nan")
    u, inv = np.unique(gid, return_inverse=True)
    n = np.bincount(inv).astype(float)
    c = np.bincount(inv, weights=correct)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(u), (B, len(u)))
    acc = c[idx].sum(1) / n[idx].sum(1)
    return float(c.sum() / n.sum()), float(np.quantile(acc, 0.025)), float(np.quantile(acc, 0.975))


def _cv_predict(X, y, groups, make, n_splits=5, seed=0):
    from sklearn.model_selection import StratifiedGroupKFold
    classes = np.unique(y)
    ns = int(max(2, min(n_splits, min(np.sum(y == c) for c in classes))))
    cv = StratifiedGroupKFold(n_splits=ns, shuffle=True, random_state=seed)
    pred = np.empty(len(y), dtype=y.dtype)
    for tr, te in cv.split(X, y, groups):
        clf = make()
        clf.fit(X[tr], y[tr])
        pred[te] = clf.predict(X[te])
    return pred


def diagnosis_methods():
    from sklearn.dummy import DummyClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    mom = ["mean", "std", "skew", "kurt", "mean_abs", "max_abs", "slope", "mean_nu_rel", "std_nu_rel"]
    return {
        "majority class": (None, lambda: DummyClassifier(strategy="most_frequent")),
        "depth-2 rule (mean, sd, |z|)": (["mean", "std", "mean_abs"], lambda: DecisionTreeClassifier(max_depth=2, random_state=0)),
        "LDA on residual moments": (mom, lambda: make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())),
        "RF on residual moments": (mom, lambda: RandomForestClassifier(300, min_samples_leaf=2, random_state=0, n_jobs=1)),
        "RF on all features": ("all", lambda: RandomForestClassifier(300, min_samples_leaf=2, random_state=0, n_jobs=1)),
    }


def diagnosis_baselines(results_dir, detectors=("D4_cusum", "D9_multiscale"), pool_by=("model", "scheme"),
                        seed=0, n_boot=1000, min_n=30):
    """Fault isolation vs baselines on the diagnosis features stored by `main`. Returns (summary_df, subgroup_df).

    summary_df  one row per (detector, pool, method): accuracy given detection (CI), macro-F1, end-to-end accuracy
                (detected AND correctly isolated / all failed patients).
    subgroup_df accuracy per true class, per failure magnitude and per structural subtype (periodic / saturation /
                delay / omitted) for every method."""
    from sklearn.metrics import f1_score
    from .analysis import load_diag
    from .diagnose import feature_names, GROUPS
    names = feature_names(GROUPS)
    methods = diagnosis_methods()
    D = load_diag(results_dir)
    rows, sub_rows = [], []
    for det in detectors:
        pools = {}
        for cid, (arr, meta) in D.items():
            if f"{det}__X" in arr:
                pools.setdefault(tuple(meta[k] for k in pool_by), []).append((cid, arr, meta))
        for key, items in pools.items():
            X = np.concatenate([a[f"{det}__X"] for _, a, _ in items])
            y = np.concatenate([a[f"{det}__kind"] for _, a, _ in items]).astype(int)
            lvl = np.concatenate([a[f"{det}__level"] for _, a, _ in items]).astype(str)
            sub = np.concatenate([a[f"{det}__sub"] for _, a, _ in items]).astype(int)
            gid = np.concatenate([[f"{cid}_{p}" for p in a[f"{det}__pid"]] for cid, a, _ in items])
            n_total = sum(int(m["n_patients"]) * 9 for _, _, m in items)
            if len(y) < min_n or len(np.unique(y)) < 2:
                continue
            for mname, (cols, make) in methods.items():
                if cols is None:
                    Xm = np.zeros((len(y), 1))
                elif cols == "all":
                    Xm = X
                else:
                    Xm = X[:, [names.index(c) for c in cols]]
                pred = _cv_predict(Xm, y, gid, make, seed=seed)
                ok = pred == y
                acc, lo, hi = _acc_ci(ok, gid, n_boot, seed)
                base = dict(zip(pool_by, key), detector=det, method=mname)
                rows.append(dict(base, n_detected=len(y), n_failed_total=n_total, accuracy_given_detected=acc,
                                 acc_lo=lo, acc_hi=hi, macro_f1=float(f1_score(y, pred, average="macro")),
                                 end_to_end_accuracy=acc * len(y) / n_total,
                                 e2e_lo=lo * len(y) / n_total, e2e_hi=hi * len(y) / n_total))
                for k in np.unique(y):
                    m = y == k
                    a, l, h = _acc_ci(ok[m], gid[m], n_boot, seed)
                    sub_rows.append(dict(base, group="true class", value=KIND_NAMES[int(k)], n=int(m.sum()), acc=a, lo=l, hi=h))
                for v in ("small", "medium", "large"):
                    m = lvl == v
                    if m.any():
                        a, l, h = _acc_ci(ok[m], gid[m], n_boot, seed)
                        sub_rows.append(dict(base, group="magnitude", value=v, n=int(m.sum()), acc=a, lo=l, hi=h))
                for s in np.unique(sub[y == 2]):
                    m = (y == 2) & (sub == s)
                    a, l, h = _acc_ci(ok[m], gid[m], n_boot, seed)
                    sub_rows.append(dict(base, group="structural subtype", value=SUB_NAMES[int(s)], n=int(m.sum()), acc=a, lo=l, hi=h))
    return pd.DataFrame(rows), pd.DataFrame(sub_rows)
