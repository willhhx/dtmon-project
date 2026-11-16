"""D11: cumulative Bayesian change-point detector, plus its confirmatory evaluation.

Idea. D7 (BOCPD) asks "did the current run start within the last few observations?". That question forgets a change
once it is older than the window. D11 keeps the same Adams-MacKay run-length filter and the same Normal-Gamma model,
with the same hyper-parameters as D7, but asks the monitoring question directly:
"has at least one change point occurred since monitoring began?". The posterior mass of all paths with NO change
since the end of burn-in is tracked exactly in a second array, so evidence accumulates over the whole window.

Protocol (fixed before the confirmatory run, do not edit after looking at results):
  * design was chosen on development seed 11 (200 patients, noise 0.1); hyper-parameters = D7 defaults, no tuning;
  * confirmatory benchmark: fresh seed (default 2026), 32 conditions, thresholds calibrated to alpha on an
    independent cohort, exactly as in the main benchmark;
  * comparison: paired by patient against the strongest baseline per model named in the paper
    (glucose: D6_sr, tumor: D7_bocpd), bootstrap 95% CI; a cell is a GAIN if delta >= 0.05 and the CI excludes 0,
    a LOSS if delta <= -0.05 and the CI excludes 0;
  * the detector is reported as an improvement for a model only if the mean paired difference over cells is
    positive with a 95% CI (bootstrap over cells) excluding 0, and its false-alarm probability stays calibrated.

Run:  python3 -m dtmon.newdet bench|transfer|real|decide   (see scripts/run_newdet.sh)
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp

from .detectors import Detector, CUSUM, ShiryaevRoberts, BOCPD, ConformalAlarm, MultiScale

STRONGEST = {"glucose": "D6_sr", "tumor": "D7_bocpd"}       # pre-specified in the paper
NEW = "D11_cumbocpd"


def _student_logpdf(x, mu, ka, al, be):
    nu = 2 * al
    s2 = be * (ka + 1) / (al * ka)
    return (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * np.pi * s2)
            - (nu + 1) / 2 * np.log1p((x - mu) ** 2 / (nu * s2)))


class CumBOCPD(Detector):
    """D11: posterior log-odds that at least one change point occurred since monitoring began.
    Same filter, model and hyper-parameters as D7; burn-in observations are fed to the filter but not scored."""
    name = NEW

    def __init__(self, hazard=1 / 100.0, rmax=50, mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0):
        self.hazard, self.rmax = hazard, rmax
        self.mu0, self.kappa0, self.alpha0, self.beta0 = mu0, kappa0, alpha0, beta0

    def stat(self, R):
        P, M = R.Z.shape
        Rm, h = self.rmax, self.hazard
        logp = np.full((P, Rm + 1), -np.inf)          # run-length posterior
        logp[:, 0] = 0.0
        logq = logp.copy()                            # part of it with no change since monitoring began
        mu = np.full((P, Rm + 1), self.mu0)
        ka = np.full((P, Rm + 1), self.kappa0)
        al = np.full((P, Rm + 1), self.alpha0)
        be = np.full((P, Rm + 1), self.beta0)
        out = np.full((P, M), np.nan)
        valid = np.isfinite(R.Z)
        Z = np.nan_to_num(R.Z)

        def shift(a, a0):                             # run length r -> r+1, last bin absorbs
            b = np.empty_like(a)
            b[:, 0] = a0
            b[:, 1:] = a[:, :-1]
            b[:, -1] = a[:, -1]
            return b

        def grow(lg):
            g = lg + np.log1p(-h)
            n = np.full_like(lg, -np.inf)
            n[:, 1:] = g[:, :-1]
            n[:, -1] = np.logaddexp(n[:, -1], g[:, -1])
            return n

        for m in range(M):
            v = valid[:, m]
            if not v.any():
                continue
            a = R.active[:, m]
            x = Z[:, m][:, None]
            lp = _student_logpdf(x, mu, ka, al, be)
            lg = logp + lp
            nl = grow(lg)
            nl[:, 0] = logsumexp(lg, axis=1) + np.log(h)          # a change point now
            norm = logsumexp(nl, axis=1, keepdims=True)
            nl = nl - norm
            nq = grow(logq + lp) - norm                           # the change-point bin gets no "no change" mass
            nq = np.where(a[:, None], nq, nl)                     # during burn-in q follows p
            vv = v[:, None]
            mu_n = (ka * mu + x) / (ka + 1)
            be_n = be + ka * (x - mu) ** 2 / (2 * (ka + 1))
            logp = np.where(vv, nl, logp)
            logq = np.where(vv, nq, logq)
            mu = np.where(vv, shift(mu_n, self.mu0), mu)
            be = np.where(vv, shift(be_n, self.beta0), be)
            ka = np.where(vv, shift(ka + 1, self.kappa0), ka)
            al = np.where(vv, shift(al + 0.5, self.alpha0), al)
            lnc = np.minimum(logsumexp(logq, axis=1), -1e-12)     # log P(no change since baseline | data)
            out[:, m] = np.log(-np.expm1(lnc)) - lnc
        return self._finish(out, R)


def detector_set():
    """Baselines kept for the comparison plus the new detector."""
    return [CUSUM(), ShiryaevRoberts(), BOCPD(), ConformalAlarm(), MultiScale(), CumBOCPD()]


# ----------------------------------------------------------------------------------------------
def _boot_mean(x, n_boot=2000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    b = x[rng.integers(0, x.size, (n_boot, x.size))].mean(1)
    return float(x.mean()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def compare(patients: pd.DataFrame, cand: str, base: str, n_boot=1000, min_effect=0.05) -> pd.DataFrame:
    """Paired comparison in every cell (condition x perturbation type x size)."""
    from .calibrate import paired_bootstrap, OUT_DET
    d = patients[patients.detector.isin([cand, base])].copy()
    d["det"] = (d.outcome == OUT_DET).astype(float)
    w = d.pivot_table(index=["model", "scheme", "noise", "cond_id", "kind", "level", "patient"],
                      columns="detector", values="det").reset_index()
    rows = []
    for key, g in w.groupby(["model", "scheme", "noise", "cond_id", "kind", "level"]):
        dl, lo, hi = paired_bootstrap(g[cand].values, g[base].values, n_boot, 0)
        r = dict(zip(["model", "scheme", "noise", "cond_id", "kind", "level"], key))
        r.update(cand=cand, base=base, det_cand=g[cand].mean(), det_base=g[base].mean(), delta=dl, lo=lo, hi=hi,
                 gain=bool(dl >= min_effect and lo > 0), loss=bool(dl <= -min_effect and hi < 0))
        rows.append(r)
    return pd.DataFrame(rows)


def cmd_bench(a):
    from .experiments import condition_grid, run_grid, load_results, load_patients
    out = os.path.join(a.out, "newdet", "bench")
    conds = condition_grid(models=a.models, noises=a.noises, schemes=a.schemes)
    run_grid(conds, out, n_jobs=a.n_jobs, n_patients=a.n_patients, seed=a.seed, alpha=a.alpha,
             detectors=detector_set(), store_diag=False, cache_dir=os.path.join(a.out, "cache"))
    S, Pt = load_results(out), load_patients(out)
    summ = (S.groupby(["model", "detector"])[["detect_rate", "far_null_full"]].mean().round(3))
    print(summ.to_string())
    cells = pd.concat([compare(Pt, NEW, b, a.n_boot) for b in ("D6_sr", "D7_bocpd")], ignore_index=True)
    cells.to_csv(os.path.join(a.out, "newdet", "bench_cells.csv"), index=False)
    S.groupby(["model", "scheme", "noise", "detector"])[["detect_rate", "far_null_full", "median_delay"]].mean() \
        .reset_index().to_csv(os.path.join(a.out, "newdet", "bench_summary.csv"), index=False)
    print(_headline(cells).round(3).to_string(index=False))


def _headline(cells):
    rows = []
    for (m, b), g in cells.groupby(["model", "base"]):
        mean, lo, hi = _boot_mean(g.delta.values)
        rows.append(dict(model=m, baseline=b, strongest=(STRONGEST.get(m) == b), cells=len(g),
                         det_new=g.det_cand.mean(), det_base=g.det_base.mean(), mean_delta=mean, lo=lo, hi=hi,
                         share_gain=g.gain.mean(), share_loss=g.loss.mean()))
    return pd.DataFrame(rows)


def cmd_transfer(a):
    from .extra import power_transfer
    out = os.path.join(a.out, "newdet")
    os.makedirs(out, exist_ok=True)
    S = []
    for m in a.models:
        _, _, s = power_transfer(m, n_patients=a.n_patients, seed=a.seed, alpha=a.alpha, n_jobs=a.n_jobs,
                                 detectors=detector_set(), cache_dir=os.path.join(a.out, "cache"))
        S.append(s)
        print(f"[newdet transfer] {m} done", flush=True)
    S = pd.concat(S, ignore_index=True)
    S.to_csv(os.path.join(out, "transfer_summary.csv"), index=False)
    print(S[["model", "shift", "detector", "far", "far_lo", "far_hi", "power_nominal_thr", "power_oracle_thr"]]
          .round(3).to_string(index=False))


def cmd_real(a):
    from .realdata import load_series, run_realdata
    out = os.path.join(a.out, "newdet")
    os.makedirs(out, exist_ok=True)
    series = load_series(a.csv, "id", "time", "value", min_obs=a.k_burn + 4)
    far, pw, info = run_realdata(series, k_burn=a.k_burn, alpha=a.alpha, n_splits=a.n_splits, seed=a.seed,
                                 detectors=detector_set())
    far.to_csv(os.path.join(out, "real_far.csv"), index=False)
    pw.to_csv(os.path.join(out, "real_power.csv"), index=False)
    print(info)
    print(far.round(3).to_string(index=False))
    print(pw.pivot_table(index="detector", columns="magnitude_sigma", values="power").round(2).to_string())


def cmd_decide(a):
    out = os.path.join(a.out, "newdet")
    cells = pd.read_csv(os.path.join(out, "bench_cells.csv"))
    summ = pd.read_csv(os.path.join(out, "bench_summary.csv"))
    head = _headline(cells)
    far = summ[summ.detector == NEW].groupby("model").far_null_full.mean()
    dec = {}
    for m, b in STRONGEST.items():
        r = head[(head.model == m) & (head.baseline == b)]
        if r.empty:
            continue
        r = r.iloc[0]
        calibrated = bool(far.get(m, np.nan) <= 2 * a.alpha)
        if r.lo > 0 and calibrated:
            verdict = "IMPROVEMENT over the strongest baseline"
        elif r.hi < 0:
            verdict = "below the strongest baseline"
        else:
            verdict = "comparable to the strongest baseline"
        dec[m] = dict(baseline=b, mean_delta=float(r.mean_delta), ci=[float(r.lo), float(r.hi)],
                      share_cells_gain=float(r.share_gain), share_cells_loss=float(r.share_loss),
                      far_new=float(far.get(m, np.nan)), verdict=verdict)
    tp = os.path.join(out, "transfer_summary.csv")
    if os.path.exists(tp):
        T = pd.read_csv(tp)
        t = T[T.detector == NEW]
        dec["misspecification"] = dict(max_far_new=float(t.far.max()),
                                       shifts_far_above_010=int((t.far > 0.10).sum()), n_shifts=int(len(t)))
        piv = T.pivot_table(index=["model", "shift"], columns="detector", values="power_oracle_thr")
        dec["misspecification"]["shifts_where_new_has_top_power"] = int((piv.idxmax(axis=1) == NEW).sum())
    with open(os.path.join(out, "decision.json"), "w") as f:
        json.dump(dec, f, indent=2)
    print(head.round(3).to_string(index=False))
    print(json.dumps(dec, indent=2))


def main(argv=None):
    p = argparse.ArgumentParser(prog="dtmon.newdet", description="D11 confirmatory evaluation")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in [("bench", cmd_bench), ("transfer", cmd_transfer), ("real", cmd_real), ("decide", cmd_decide)]:
        sp = sub.add_parser(name)
        sp.add_argument("--out", default="results")
        sp.add_argument("--seed", type=int, default=2026)
        sp.add_argument("--alpha", type=float, default=0.05)
        sp.add_argument("--n-patients", type=int, default=500)
        sp.add_argument("--n-jobs", type=int, default=1)
        sp.add_argument("--n-boot", type=int, default=1000)
        sp.add_argument("--models", nargs="+", default=["glucose", "tumor"])
        sp.add_argument("--noises", nargs="+", type=float, default=[0.05, 0.1, 0.2, 0.4])
        sp.add_argument("--schemes", nargs="+", default=["dense", "moderate", "sparse", "realistic"])
        sp.add_argument("--csv", default="data/tumor.csv")
        sp.add_argument("--k-burn", type=int, default=5)
        sp.add_argument("--n-splits", type=int, default=100)
        sp.set_defaults(fn=fn)
    a = p.parse_args(argv)
    t0 = time.time()
    a.fn(a)
    print(f"[newdet {a.cmd}] finished in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
