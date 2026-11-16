"""Real-data validation (semi-synthetic): does the benchmark's picture survive on REAL longitudinal tumour data?

Input: a long-format csv with columns id, time, value (tumour volume or any positive, growth-like biomarker;
one row per measurement). Public sources to try (verify licence/format yourself): mouse tumour-growth curves from
Benzekry et al. 2014 (PLoS Comput Biol 10:e1003800) or the human tumour-growth benchmark released with
Ghaffari Laleh et al. (bioRxiv 2021.10.23.465549). Any csv with these three columns works
(`--id-col/--time-col/--value-col` rename them).

What it does (honest scope: there is no ground-truth twin failure in real data)
  1. Frozen per-patient twin: a Gompertz growth curve fitted on log(value) to the first `k_burn` observations
     (the twin sees nothing after that). Standardised residual z_t = (log y_t - log yhat_t) / sigma_i with
     sigma_i shrunk towards the pooled burn-in residual scale.  These residuals are NOT iid N(0,1) (extrapolation
     error), exactly the situation in which nominal guarantees fail.
  2. Calibration check on real data: thresholds calibrated on a random half of the patients, false-alarm rate on
     the other half, repeated over random splits. Also the NOMINAL Shiryaev-Roberts threshold log(N/alpha) from the
     theory (T1) without any calibration.
  3. Injected failures on the held-out half (the series is corrupted after an onset in the middle of the monitored
     window): level shift, extra variance, drifting trend; magnitudes in units of sigma_i. Power at the calibrated
     threshold, per detector.
Detectors are the same classes as in the simulation study, run on observation-index time.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .detectors import Residuals, all_detectors, ShiryaevRoberts
from .calibrate import calibrate_threshold, evaluate, per_patient_max, OUT_DET
from .extra import wilson


# ----------------------------------------------------------------------------------------------
def load_series(path, id_col="id", time_col="time", value_col="value", min_obs=10):
    df = pd.read_csv(path)
    df = df[[id_col, time_col, value_col]].dropna()
    df = df[df[value_col] > 0]
    out = {}
    for pid, g in df.groupby(id_col):
        g = g.sort_values(time_col)
        t = g[time_col].to_numpy(float)
        y = g[value_col].to_numpy(float)
        keep = np.r_[True, np.diff(t) > 0]
        t, y = t[keep], y[keep]
        if t.size >= min_obs:
            out[pid] = (t - t[0], y)
    return out


def gompertz_log(t, p):
    """log V(t) = logV0 + (a/b)(1 - exp(-b t)); b -> 0 is exponential growth."""
    logv0, a, b = p
    b = np.where(np.abs(b) < 1e-8, 1e-8, b)
    return logv0 + (a / b) * (1.0 - np.exp(-b * t))


def fit_twin(t, logy, prior=None):
    """MAP Gompertz fit on the burn-in window. `prior` = (a_mu, a_sd, logb_mu, logb_sd, noise_sd) shrinks the growth
    parameters towards the population (empirical Bayes); without it the fit is plain least squares.
    Returns (params, in-sample residual sd, ok)."""
    span = max(t.max() - t.min(), 1e-6)
    lb = np.array([logy.min() - 3, -20.0 / span, 1e-6])
    ub = np.array([logy.max() + 3, 20.0 / span, 30.0 / span])
    slope = max((logy[-1] - logy[0]) / span, 1e-3)

    def fun(p):
        r = gompertz_log(t, p) - logy
        if prior is None:
            return r
        a_mu, a_sd, lb_mu, lb_sd, s = prior
        return np.r_[r / s, (p[1] - a_mu) / a_sd, (np.log(p[2]) - lb_mu) / lb_sd]
    starts = [np.array([logy[0], slope * (1 + b0 * span / 2), b0]) for b0 in (0.05, 0.3, 1.0, 3.0)]
    if prior is not None:
        starts.append(np.array([logy[0], prior[0], np.exp(prior[2])]))
    best = None
    for p0 in starts:
        p0 = np.clip(p0, lb + 1e-9, ub - 1e-9)
        try:
            sol = least_squares(fun, p0, bounds=(lb, ub))
        except Exception:
            continue
        if best is None or sol.cost < best.cost:
            best = sol
    if best is None:
        return starts[0], float("nan"), False
    res = gompertz_log(t, best.x) - logy
    dof = max(t.size - 3, 1)
    return best.x, float(np.sqrt(np.sum(res ** 2) / dof)), bool(best.success)


def build_twins(series, k_burn=5, n0=3.0, shrink=True):
    """Frozen twins for every patient with >= k_burn + 4 observations. Pass 1: per-patient least squares; pass 2
    (shrink=True): MAP refit with an empirical-Bayes prior on (a, log b) estimated from pass 1 (robust median/MAD),
    which stabilises extrapolation from few burn-in points. Returns a list of dicts."""
    base = []
    for pid, (t, y) in series.items():
        if t.size < k_burn + 4:
            continue
        logy = np.log(y)
        p, s, ok = fit_twin(t[:k_burn], logy[:k_burn])
        if ok and np.isfinite(s):
            base.append((pid, t, logy, p, s))
    if not base:
        raise ValueError("no usable patients: need >= k_burn+4 observations and a successful fit")
    s_pool = float(np.sqrt(np.median([b[4] ** 2 for b in base])))
    s_pool = max(s_pool, 0.02)
    prior = None
    if shrink and len(base) >= 10:
        A = np.array([b[3][1] for b in base])
        LB = np.log(np.array([b[3][2] for b in base]))
        mad = lambda x: 1.4826 * np.median(np.abs(x - np.median(x)))
        prior = (float(np.median(A)), max(float(mad(A)), 1e-3), float(np.median(LB)), max(float(mad(LB)), 0.1), s_pool)
    twins, s2 = [], []
    for pid, t, logy, p, s in base:
        if prior is not None:
            p, s, ok = fit_twin(t[:k_burn], logy[:k_burn], prior)
            if not ok or not np.isfinite(s):
                continue
        twins.append(dict(id=pid, t=t, logy=logy, p=p, s=s))
        s2.append(s ** 2)
    s2_pool = float(np.median(s2))
    dof = max(k_burn - 3, 1)
    for tw in twins:
        tw["sigma"] = float(np.sqrt((dof * tw["s"] ** 2 + n0 * s2_pool) / (dof + n0)))
        tw["sigma"] = max(tw["sigma"], 0.02)
    return twins


def innovations(tw, logy, q_frac=1.0):
    """State-corrected twin residuals (the real-data analogue of the EKF twin in the simulation study): the frozen
    Gompertz curve plus a random-walk offset x_t tracked by a scalar Kalman filter. Process sd per median visit gap
    is q_frac * sigma_i (q_frac=1 gave ~N(0,1) innovations on well-specified synthetic series). Returns standardised innovations z_k = (r_k - xhat_k|k-1) / sqrt(P_k|k-1 + sigma_i^2),
    which are ~N(0,1) and white when the twin is valid and slow model error is absorbed by the offset."""
    t = tw["t"]
    r = logy - gompertz_log(t, tw["p"])
    sig2 = tw["sigma"] ** 2
    gap = max(float(np.median(np.diff(t))), 1e-6)
    x, Pc = 0.0, sig2
    z = np.empty(len(t))
    for k in range(len(t)):
        if k > 0:
            Pc = Pc + (q_frac ** 2) * sig2 * (t[k] - t[k - 1]) / gap
        S = Pc + sig2
        z[k] = (r[k] - x) / np.sqrt(S)
        K = Pc / S
        x = x + K * (r[k] - x)
        Pc = (1 - K) * Pc
    return z


def residuals_from_twins(twins, logy_list, k_burn, q_frac=1.0):
    """Residuals object on observation-index time (burn-in = first k_burn observations)."""
    P = len(twins)
    M = max(len(tw["t"]) for tw in twins)
    Z = np.full((P, M), np.nan)
    Tm = np.full((P, M), np.inf)
    for i, (tw, ly) in enumerate(zip(twins, logy_list)):
        n = len(tw["t"])
        Z[i, :n] = innovations(tw, ly, q_frac)
        Tm[i, :n] = np.arange(n, dtype=float)
    Z = np.clip(Z, -8, 8)
    valid = np.isfinite(Z)
    active = valid & (Tm >= k_burn)
    cal = np.full((P, k_burn), np.nan)
    cal[:] = np.abs(Z[:, :k_burn])
    return Residuals(Z, Tm, active, cal, float(k_burn), np.full(P, 6.0))


def inject(tw, kind, mag, rng, onset_idx):
    """Corrupt the observed log series after onset. mag in units of the patient's sigma."""
    ly = tw["logy"].copy()
    n = ly.size
    m = n - onset_idx
    sig = tw["sigma"]
    if kind == "shift":
        ramp = np.minimum(np.arange(1, m + 1) / 3.0, 1.0)
        ly[onset_idx:] += mag * sig * ramp
    elif kind == "variance":
        ly[onset_idx:] += rng.normal(0.0, mag * sig, m)
    elif kind == "trend":
        ly[onset_idx:] += mag * sig * np.arange(1, m + 1) / max(m, 1) * 2.0
    else:
        raise ValueError(kind)
    return ly


def _cluster(records, P, rng, n_boot):
    """records: list of (mouse_index_array, 0/1 flags). Estimate = pooled rate; CI = bootstrap over MICE (not over
    split x mouse evaluations, which re-use the same few mice)."""
    sm, ct = np.zeros(P), np.zeros(P)
    for idx, fl in records:
        np.add.at(sm, idx, fl); np.add.at(ct, idx, 1.0)
    est = sm.sum() / ct.sum()
    ok = np.where(ct > 0)[0]
    bs = np.empty(n_boot)
    for b in range(n_boot):
        j = rng.choice(ok, ok.size, replace=True)
        bs[b] = sm[j].sum() / ct[j].sum()
    return est, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def run_realdata(series, k_burn=5, alpha=0.05, n_splits=20, seed=0, kinds=("shift", "variance", "trend"),
                 mags=(3.0, 6.0, 12.0), detectors=None, q_frac=1.0, n_boot=2000):
    """Returns (far_df, power_df, info). far_df has one row per detector (+ the nominal-theory SR threshold)."""
    twins = build_twins(series, k_burn)
    P = len(twins)
    if P < 20:
        raise ValueError(f"only {P} usable patients; need >= 20 for a split-sample check")
    rng = np.random.default_rng(seed)
    onset = np.array([k_burn + max(2, (len(tw["t"]) - k_burn) // 2) for tw in twins])
    clean = residuals_from_twins(twins, [tw["logy"] for tw in twins], k_burn, q_frac)
    dets_proto = detectors if detectors is not None else all_detectors()
    far_acc, pw_acc = {}, {}
    n_far = 0
    sr_nom = []
    for sp in range(n_splits):
        perm = rng.permutation(P)
        ical, iev = perm[: P // 2], perm[P // 2:]

        def sub(R, idx):
            return Residuals(R.Z[idx], R.Tm[idx], R.active[idx], R.cal[idx], R.burn_in, R.period[idx])
        Rc, Re = sub(clean, ical), sub(clean, iev)
        from copy import deepcopy
        dets = [deepcopy(d) for d in dets_proto]
        thr = {}
        for d in dets:
            d.fit(Rc)
            thr[d.name] = calibrate_threshold(d.stat_cached(Rc), Rc.Tm, alpha)
        for d in dets:
            mx = per_patient_max(d.stat_cached(Re), Re.Tm)
            far_acc.setdefault(d.name, []).append((iev, (mx > thr[d.name]).astype(float)))
            if isinstance(d, ShiryaevRoberts):
                N = float(np.mean(Re.active.sum(1)))
                h_nom = np.log(N / alpha)
                sr_nom.append((iev, (mx > h_nom).astype(float)))
        for kind in kinds:
            for mg in mags:
                lys = [inject(twins[i], kind, mg, rng, int(onset[i])) for i in iev]
                Rf = residuals_from_twins([twins[i] for i in iev], lys, k_burn, q_frac)
                for d in dets:
                    ev = evaluate(d.stat_cached(Rf), Rf.Tm, thr[d.name], onset[iev].astype(float), float(Rf.M))
                    pw_acc.setdefault((d.name, kind, mg), []).append((iev, (ev["outcome"] == OUT_DET).astype(float)))
    brng = np.random.default_rng(seed + 1)
    far_rows = []
    for name, lst in far_acc.items():
        est, lo, hi = _cluster(lst, P, brng, n_boot)
        far_rows.append(dict(detector=name, threshold="calibrated on real calibration split", far=est, far_lo=lo,
                             far_hi=hi, alpha=alpha, n_eval=int(sum(len(i) for i, _ in lst))))
    if sr_nom:
        est, lo, hi = _cluster(sr_nom, P, brng, n_boot)
        far_rows.append(dict(detector="D6_sr", threshold="nominal theory log(N/alpha) (T1)", far=est, far_lo=lo,
                             far_hi=hi, alpha=alpha, n_eval=int(sum(len(i) for i, _ in sr_nom))))
    pw_rows = []
    for (name, kind, mg), lst in pw_acc.items():
        est, lo, hi = _cluster(lst, P, brng, n_boot)
        pw_rows.append(dict(detector=name, injected=kind, magnitude_sigma=mg, power=est, power_lo=lo, power_hi=hi))
    info = dict(n_patients=P, k_burn=k_burn, median_obs=float(np.median([len(tw["t"]) for tw in twins])),
                n_splits=n_splits, clean_innov_sd=float(np.nanstd(clean.Z[clean.active])),
                n_boot=n_boot, ci="bootstrap over mice")
    if info["clean_innov_sd"] > 2.0:
        print(f"[realdata] WARNING: clean innovation sd = {info['clean_innov_sd']:.2f} (>2): twin badly mis-scaled "
              f"(burn-in too short?); power results are not interpretable. Increase --k-burn.")
    return pd.DataFrame(far_rows), pd.DataFrame(pw_rows), info


def make_demo_csv(path, n_patients=80, seed=0, noise=0.12):
    """Synthetic Gompertz tumour-like series for testing the pipeline end to end (NOT real data)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_patients):
        n = rng.integers(12, 20)
        t = np.sort(rng.uniform(0, 40, n))
        p = np.array([np.log(rng.uniform(20, 80)), rng.uniform(0.15, 0.3), rng.uniform(0.02, 0.06)])
        y = np.exp(gompertz_log(t, p) + rng.normal(0, noise, n))
        rows += [(f"m{i}", float(tt), float(v)) for tt, v in zip(t, y)]
    pd.DataFrame(rows, columns=["id", "time", "value"]).to_csv(path, index=False)
    return path
