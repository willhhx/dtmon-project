"""Experiment orchestration.

One *condition* = (model, noise CV, sampling scheme, missingness, calibration error, filter variant).
For each condition we:
  1. simulate an independent no-failure CALIBRATION cohort, run the twin filter, fit detectors and
     calibrate every detector's threshold to the same false-alarm probability alpha;
  2. simulate an EVALUATION population and, on the SAME patients (common random numbers), a no-failure
     cohort plus failure cohorts (drift / structure / feedback x magnitude levels);
  3. record per-patient outcomes (alarm time, delay, score) for each detector, AUROC vs the paired null
     cohort, and diagnosis features at each detector's alarm.
Results are written as csv.gz / npz so that figures and statistics can be regenerated without re-simulating.
"""
from __future__ import annotations
import os, time, json, hashlib, itertools
import numpy as np
import pandas as pd

from .models import get_model, KIND_NAMES, SUB_NAMES
from .simulate import make_population, make_fault, simulate_truth, observe
from .filter import run_ekf
from .detectors import prepare_residuals, all_detectors, ablation_detectors
from .calibrate import (calibrate_threshold, evaluate, summarize, auroc, far_full, OUT_DET)
from .diagnose import extract_features, extract_sequences, feature_names, GROUPS

SCHEME_ORDER = ["dense", "moderate", "sparse", "realistic"]
FAULT_KINDS = (1, 2, 3)
LEVELS = ("small", "medium", "large")
DIAG_DETECTORS = ("D4_cusum", "D9_multiscale", "D10_hybrid")


def _truth(model, pop, kind, level, sub, pop_seed, P, cache_dir):
    """Truth simulation with optional on-disk cache (truth does not depend on noise/sampling)."""
    key = f"{model.name}_{pop_seed}_{P}_{kind}_{level}_{sub}"
    path = os.path.join(cache_dir, key + ".npy") if cache_dir else None
    fault = make_fault(model, pop, kind, level if kind else "medium", sub=sub, seed=pop_seed)
    if path and os.path.exists(path):
        return np.load(path), fault
    y = simulate_truth(pop, fault)
    if path:
        os.makedirs(cache_dir, exist_ok=True)
        np.save(path, y)
    return y, fault


def _residuals(model, pop, cohort, mode, normalize, q_frac, p0_frac, cv_assumed_scale, clip=8.0):
    out = run_ekf(model, pop, cohort, mode=mode, q_frac=q_frac, p0_frac=p0_frac,
                  cv_assumed_scale=cv_assumed_scale)
    aux = dict(corr=out.corr, infl=out.infl, nu_rel=out.nu_rel)
    return prepare_residuals(out.z, cohort.mask, cohort.t, model.burn_in, pop.periods,
                             normalize=normalize, clip=clip, aux_grid=aux)


def run_condition(cond: dict, n_patients: int = 100, seed: int = 0, alpha: float = 0.05,
                  levels=LEVELS, kinds=FAULT_KINDS, detectors=None, with_ablation_detectors=False,
                  cache_dir: str | None = None, diag_groups=GROUPS, store_diag=True, verbose=True):
    """Run one condition. Returns dict(summary=DataFrame, patients=DataFrame, diag=dict, thresholds=dict)."""
    t_start = time.time()
    model = get_model(cond["model"])
    noise = cond.get("noise", 0.1)
    scheme = cond.get("scheme", "dense")
    miss = cond.get("miss", 0.0)
    miss_kind = cond.get("miss_kind", "random")
    cal_sigma = cond.get("cal_sigma", 0.05)
    mode = cond.get("mode", "ekf")
    normalize = cond.get("normalize", True)
    q_frac = cond.get("q_frac", 0.03)
    p0_frac = cond.get("p0_frac", 0.05)
    cvs = cond.get("cv_assumed_scale", 1.0)
    cv_scale = cond.get("cv_scale", 1.0)             # patient-parameter heterogeneity
    freq_cv = cond.get("freq_cv", 0.0)
    treat_frac = cond.get("treat_frac", 0.5)

    obs_seed = seed + 3000 + int(round(noise * 1000)) + 7 * SCHEME_ORDER.index(scheme)
    P = n_patients
    dets = detectors if detectors is not None else all_detectors()
    if with_ablation_detectors:
        dets = list(dets) + ablation_detectors()

    # ---- 1. calibration cohort (no failure) --------------------------------------------------
    cal_seed = seed + 1000
    pop_c = make_population(model, P, cal_seed, cv_scale, cal_sigma, p0_frac, treat_frac)
    yc, f0 = _truth(model, pop_c, 0, "medium", 0, cal_seed, P, cache_dir)
    co = observe(model, pop_c, yc, f0, scheme, noise, obs_seed + 11, miss, miss_kind, freq_cv)
    R_cal = _residuals(model, pop_c, co, mode, normalize, q_frac, p0_frac, cvs)
    thr, fit_time = {}, {}
    for d in dets:
        t0 = time.time()
        d.fit(R_cal)
        s = d.stat_cached(R_cal)
        thr[d.name] = calibrate_threshold(s, R_cal.Tm, alpha)
        fit_time[d.name] = time.time() - t0
    del R_cal

    # ---- 2. evaluation cohorts (same patients, CRN) ---------------------------------------------
    ev_seed = seed + 2000
    pop_e = make_population(model, P, ev_seed, cv_scale, cal_sigma, p0_frac, treat_frac)
    cohorts = []
    y0, f0e = _truth(model, pop_e, 0, "medium", 0, ev_seed, P, cache_dir)
    cohorts.append(("null", 0, "none", f0e, y0))
    for kind in kinds:
        for lv in levels:
            y, f = _truth(model, pop_e, kind, lv, "random", ev_seed, P, cache_dir)
            cohorts.append((KIND_NAMES[kind], kind, lv, f, y))

    stats_null = None
    summary_rows, patient_rows = [], []
    diag = {dn: [] for dn in DIAG_DETECTORS}
    seq_diag = {dn: [] for dn in DIAG_DETECTORS}
    horizon = model.horizon
    far_null = {}
    null_cache = {}
    for name, kind, lv, fault, ytrue in cohorts:
        co = observe(model, pop_e, ytrue, fault, scheme, noise, obs_seed + 22, miss, miss_kind, freq_cv)
        R = _residuals(model, pop_e, co, mode, normalize, q_frac, p0_frac, cvs)
        n_obs_post = np.array([np.sum((R.Tm[i] >= fault.onset[i]) & np.isfinite(R.Z[i])) if kind else
                               np.sum((R.Tm[i] >= model.t_change) & np.isfinite(R.Z[i])) for i in range(P)])
        if kind == 0:
            for d in dets:
                null_cache[d.name] = (d.stat_cached(R).copy(), R.Tm.copy())
                far_null[d.name] = far_full(null_cache[d.name][0], R.Tm, thr[d.name])
            del R
            continue
        for d in dets:
            t0 = time.time()
            s = d.stat_cached(R)
            ev = evaluate(s, R.Tm, thr[d.name], fault.onset, horizon)
            sm = summarize(ev)
            s_null, Tm_null = null_cache[d.name]
            ev_null = evaluate(s_null, Tm_null, thr[d.name], fault.onset, horizon)
            sm["auroc"] = auroc(ev["score_post"], ev_null["score_post"])
            sm["far_null_pre"] = float((ev_null["outcome"] == 1).mean())
            sm["far_null_full"] = far_null[d.name]
            sm["detector_time_s"] = time.time() - t0 + fit_time.get(d.name, 0.0) / (len(cohorts) - 1)
            sm["threshold"] = thr[d.name]
            sm.update(dict(model=model.name, noise=noise, scheme=scheme, miss=miss, cal_sigma=cal_sigma,
                           mode=mode, normalize=normalize, variant=cond.get("variant", "full"),
                           alpha=alpha, kind=name, level=lv, detector=d.name, n_patients=P,
                           mean_obs_post=float(n_obs_post.mean())))
            summary_rows.append(sm)
            oc = ev["outcome"]
            for i in range(P):
                patient_rows.append((d.name, name, lv, i, int(fault.sub[i]), oc[i], ev["t_alarm"][i],
                                     ev["delay"][i], ev["score_post"][i], ev_null["score_post"][i],
                                     ev_null["outcome"][i], fault.onset[i], n_obs_post[i]))
            if store_diag and d.name in DIAG_DETECTORS:
                X, ok = extract_features(R, ev["alarm_idx"], groups=diag_groups)
                Sq, ok2 = extract_sequences(R, ev["alarm_idx"])
                det_mask = (oc == OUT_DET) & ok
                diag[d.name].append(dict(X=X[det_mask], pid=np.nonzero(det_mask)[0], kind=np.full(det_mask.sum(), kind),
                                         level=np.full(det_mask.sum(), lv), sub=fault.sub[det_mask],
                                         seq=Sq[det_mask], n_total=P))
        del R
    summary = pd.DataFrame(summary_rows)
    patients = pd.DataFrame(patient_rows, columns=["detector", "kind", "level", "patient", "sub", "outcome",
                                                   "t_alarm", "delay", "score_post", "score_null_post",
                                                   "outcome_null", "onset", "n_obs_post"])
    for c, v in dict(model=model.name, noise=noise, scheme=scheme, miss=miss, cal_sigma=cal_sigma,
                     variant=cond.get("variant", "full")).items():
        patients[c] = v
    if verbose:
        print(f"  [{model.name} noise={noise} scheme={scheme} miss={miss} var={cond.get('variant', 'full')}] "
              f"done in {time.time() - t_start:.0f}s", flush=True)
    return dict(summary=summary, patients=patients, diag=diag, thresholds=thr, far_null=far_null)


# ----------------------------------------------------------------------------------------------
def condition_grid(models=("glucose", "tumor"), noises=(0.05, 0.1, 0.2, 0.4),
                   schemes=SCHEME_ORDER, variants=("full",), miss=(0.0,), cal_sigmas=(0.05,),
                   cv_assumed=(1.0,)):
    var_cfg = {"full": dict(mode="ekf", normalize=True),
               "no_norm": dict(mode="ekf", normalize=False),
               "open_loop": dict(mode="open_loop", normalize=True)}
    conds = []
    for m, nz, sc, v, ms, cs, ca in itertools.product(models, noises, schemes, variants, miss, cal_sigmas, cv_assumed):
        c = dict(model=m, noise=nz, scheme=sc, variant=v, miss=ms, cal_sigma=cs, cv_assumed_scale=ca)
        c.update(var_cfg[v])
        conds.append(c)
    return conds


def cond_id(c):
    s = json.dumps(c, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:10]


def save_result(res, out_dir, cid):
    os.makedirs(out_dir, exist_ok=True)
    res["summary"].to_csv(os.path.join(out_dir, f"summary_{cid}.csv.gz"), index=False)
    res["patients"].to_csv(os.path.join(out_dir, f"patients_{cid}.csv.gz"), index=False)
    flat = {}
    for dn, lst in res["diag"].items():
        if not lst:
            continue
        flat[f"{dn}__X"] = np.concatenate([a["X"] for a in lst])
        flat[f"{dn}__pid"] = np.concatenate([a["pid"] for a in lst])
        flat[f"{dn}__kind"] = np.concatenate([a["kind"] for a in lst])
        flat[f"{dn}__level"] = np.concatenate([a["level"] for a in lst])
        flat[f"{dn}__sub"] = np.concatenate([a["sub"] for a in lst])
        flat[f"{dn}__seq"] = np.concatenate([a["seq"] for a in lst])
    np.savez_compressed(os.path.join(out_dir, f"diag_{cid}.npz"), **flat)
    with open(os.path.join(out_dir, f"thr_{cid}.json"), "w") as f:
        json.dump(dict(thresholds=res["thresholds"], far_null=res["far_null"]), f)


def _run_and_save(c, out_dir, kw):
    cid = cond_id(dict(c, **{k: v for k, v in kw.items() if k in ("n_patients", "seed", "alpha")}))
    marker = os.path.join(out_dir, f"summary_{cid}.csv.gz")
    if os.path.exists(marker):
        return cid                                  # resume support
    res = run_condition(c, **kw)
    res["summary"]["cond_id"] = cid
    res["patients"]["cond_id"] = cid
    save_result(res, out_dir, cid)
    return cid


def run_grid(conds, out_dir, n_jobs=1, **kw):
    """Run conditions (optionally in parallel, resumable). kw are passed to run_condition."""
    os.makedirs(out_dir, exist_ok=True)
    if n_jobs == 1:
        return [_run_and_save(c, out_dir, kw) for c in conds]
    from joblib import Parallel, delayed
    return Parallel(n_jobs=n_jobs)(delayed(_run_and_save)(c, out_dir, kw) for c in conds)


def load_results(out_dir):
    """Concatenate summaries (and optionally patient tables) from a results directory."""
    S = [pd.read_csv(os.path.join(out_dir, f)) for f in sorted(os.listdir(out_dir)) if f.startswith("summary_")]
    return pd.concat(S, ignore_index=True) if S else pd.DataFrame()


def load_patients(out_dir):
    S = [pd.read_csv(os.path.join(out_dir, f)) for f in sorted(os.listdir(out_dir)) if f.startswith("patients_")]
    return pd.concat(S, ignore_index=True) if S else pd.DataFrame()


# ----------------------------------------------------------------------------------------------
def transfer_far(model_name: str, cal_cond: dict, test_conds: list, n_patients: int = 200, seed: int = 0,
                 alpha: float = 0.05, detectors=None, cache_dir: str | None = None):
    """Hypothesis H3 / stress test: calibrate thresholds under NOMINAL conditions (cal_cond), then measure the
    full-window false-alarm probability on no-failure cohorts generated under SHIFTED conditions (noise
    misspecification, parameter uncertainty, missing data, different sampling). A detector whose false-alarm
    rate stays near alpha is robust to that shift; conformal/per-patient normalisation should help."""
    model = get_model(model_name)
    dets = detectors if detectors is not None else all_detectors()
    P = n_patients

    def null_R(cond, pop_seed, tag):
        pop = make_population(model, P, pop_seed, cond.get("cv_scale", 1.0), cond.get("cal_sigma", 0.05),
                              cond.get("p0_frac", 0.05), cond.get("treat_frac", 0.5))
        y, f = _truth(model, pop, 0, "medium", 0, pop_seed, P, cache_dir)
        co = observe(model, pop, y, f, cond.get("scheme", "dense"), cond.get("noise", 0.1), seed + tag,
                     cond.get("miss", 0.0), cond.get("miss_kind", "random"), cond.get("freq_cv", 0.0))
        return _residuals(model, pop, co, cond.get("mode", "ekf"), cond.get("normalize", True),
                          cond.get("q_frac", 0.03), cond.get("p0_frac", 0.05), cond.get("cv_assumed_scale", 1.0))

    R = null_R(cal_cond, seed + 1000, 1)
    thr = {}
    for d in dets:
        d.fit(R)
        thr[d.name] = calibrate_threshold(d.stat_cached(R), R.Tm, alpha)
    del R
    rows = []
    for j, tc in enumerate(test_conds):
        R = null_R(tc, seed + 5000 + j, 100 + j)
        for d in dets:
            s = d.stat_cached(R)
            rows.append(dict(model=model_name, detector=d.name, far=far_full(s, R.Tm, thr[d.name]), alpha=alpha,
                             **{f"test_{k}": v for k, v in tc.items()}))
        del R
    return pd.DataFrame(rows)
