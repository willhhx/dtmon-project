"""Theoretical results with numerical verification.

PROVED here (see docs/theory.md for proofs):
  T1  finite-horizon false-alarm bound for Shiryaev-Roberts:  P0(alarm within N obs) <= N / A
  T1b robust version when the null residual law is only approximately N(0,1):
      P(alarm within N) <= N (1+eps)^N / A,  eps = sup E[Lambda | past] - 1
  T2  split-conformal p-value validity (exchangeability) and a union-bound alarm rule
  T3  comparison: mixture-SR (average of K SR recursions) alarms no later than component-j SR with threshold K*A
CITED (not re-proved): first-order expected delay  E[delay] ~ log A / KL  (Lai 1998; Lorden 1971; Pollak 1985).
ELEMENTARY: mean-shift power  Phi(delta sqrt(n) - z_{1-alpha}).
All results assume the standardised residuals are independent N(0,1) under no failure; the experiments
measure how far EKF innovations of the nonlinear ODE twins are from that assumption.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import norm
from .detectors import Residuals, ShiryaevRoberts, CUSUM


# ------------------------------------------------------------------ formulas
def sr_fa_bound(N: int, A: float) -> float:
    return float(min(1.0, N / A))


def sr_threshold_for_fa(N: int, alpha: float) -> float:
    """log-threshold log A such that P0(alarm within N observations) <= alpha."""
    return float(np.log(N / alpha))


def sr_robust_fa_bound(N: int, A: float, eps: float) -> float:
    logb = np.log(N) + N * np.log1p(eps) - np.log(A)       # computed in logs to avoid overflow
    return float(np.exp(min(logb, 0.0)))


def kl_mean(theta: float) -> float:
    return theta ** 2 / 2


def kl_var(sd: float) -> float:
    t2 = sd ** 2
    return 0.5 * (t2 - 1 - np.log(t2))


def delay_first_order(logA: float, kl: float, n_components: int = 1) -> float:
    """(log A + log K) / KL  observations (upper-bound form for a K-component mixture SR)."""
    return (logA + np.log(n_components)) / kl


def glr_power_mean(delta: float, n: int, alpha: float) -> float:
    return float(1 - norm.cdf(norm.ppf(1 - alpha) - delta * np.sqrt(n)))


def n_required(delta: float, alpha: float, power: float) -> float:
    return float(((norm.ppf(1 - alpha) + norm.ppf(power)) / delta) ** 2)


def conformal_pvalue(cal: np.ndarray, test: np.ndarray) -> np.ndarray:
    cal = np.asarray(cal)
    return (1 + (cal[None, :] >= np.asarray(test)[:, None]).sum(1)) / (len(cal) + 1)


def estimate_epsilon(Z_null: np.ndarray, sr: ShiryaevRoberts | None = None) -> float:
    """eps_hat = max(0, mean Lambda(z) - 1) over pooled null residuals."""
    sr = sr or ShiryaevRoberts()
    z = Z_null[np.isfinite(Z_null)]
    lam = np.exp(sr.log_lambdas(z))                 # (K, n)
    return float(max(0.0, lam.mean(axis=1).max() - 1.0))   # eps = max_j E[L_j] - 1


# ------------------------------------------------------------------ numerical verification
def _residuals_from_matrix(Z, burn=0):
    P, M = Z.shape
    Tm = np.tile(np.arange(M, dtype=float), (P, 1))
    active = Tm >= burn
    return Residuals(Z, Tm, active, np.abs(Z[:, :max(burn, 1)]), float(burn), np.ones(P))


def check_sr_false_alarm(N=200, A_list=(20, 50, 100, 400, 1000), n_trials=4000, seed=0):
    """T1: empirical P(alarm within N) for iid N(0,1) vs the bound N/A."""
    rng = np.random.default_rng(seed)
    R = _residuals_from_matrix(rng.normal(size=(n_trials, N)))
    s = ShiryaevRoberts().stat(R)
    mx = np.nanmax(s, axis=1)
    return [dict(N=N, A=A, empirical=float(np.mean(mx >= np.log(A))), bound=sr_fa_bound(N, A),
                 holds=bool(np.mean(mx >= np.log(A)) <= sr_fa_bound(N, A) + 3 * np.sqrt(0.25 / n_trials)))
            for A in A_list]


def check_sr_delay(thetas=(0.75, 1.5), logA=np.log(500), n_trials=3000, N=400, seed=1):
    """T3 + cited first-order delay: measured mean delay vs (logA + log K)/KL for a mean shift."""
    rng = np.random.default_rng(seed)
    sr = ShiryaevRoberts()
    K = sr.K
    out = []
    for th in thetas:
        Z = rng.normal(size=(n_trials, N)) + th
        R = _residuals_from_matrix(Z)
        s = sr.stat(R)
        over = s >= logA
        idx = np.where(over.any(1), over.argmax(1), np.nan)
        out.append(dict(theta=th, measured_mean_delay=float(np.nanmean(idx)),
                        predicted_upper=delay_first_order(logA, kl_mean(th), K),
                        predicted_single=delay_first_order(logA, kl_mean(th), 1)))
    return out


def check_power(deltas=(0.5, 1.0), n=25, alpha=0.05, n_trials=20000, seed=2):
    """Mean-shift z-test power vs the closed form Phi(delta sqrt(n) - z)."""
    rng = np.random.default_rng(seed)
    res = []
    for d in deltas:
        z = (rng.normal(size=(n_trials, n)) + d).mean(1) * np.sqrt(n)
        res.append(dict(delta=d, n=n, empirical=float(np.mean(z > norm.ppf(1 - alpha))),
                        theory=glr_power_mean(d, n, alpha)))
    return res


def check_conformal_validity(n_cal=30, n_test=20000, levels=(0.05, 0.1, 0.2), n_rep=200, seed=3):
    """T2: P(p <= u) <= u for exchangeable data (heavy-tailed to avoid Gaussian luck)."""
    rng = np.random.default_rng(seed)
    out = {u: [] for u in levels}
    for _ in range(n_rep):
        cal = np.abs(rng.standard_t(3, n_cal))
        test = np.abs(rng.standard_t(3, n_test // n_rep))
        p = conformal_pvalue(cal, test)
        for u in levels:
            out[u].append(np.mean(p <= u))
    return [dict(level=u, empirical=float(np.mean(v)), valid=bool(np.mean(v) <= u + 0.01)) for u, v in out.items()]


def check_ekf_innovations(model_name="glucose", n_patients=100, noise=0.1, cal_sigma=0.05, scheme="moderate", seed=0):
    """How far are EKF innovations from iid N(0,1) (the assumption behind T1)?  Returns the SR false-alarm
    frequency at the T1 threshold vs the nominal bound, plus eps_hat and the robust bound."""
    from .models import get_model
    from .simulate import make_population, make_fault, simulate_truth, observe
    from .filter import run_ekf
    from .detectors import prepare_residuals
    m = get_model(model_name)
    pop = make_population(m, n_patients, seed, cal_sigma=cal_sigma)
    f = make_fault(m, pop, 0)
    co = observe(m, pop, simulate_truth(pop, f), f, scheme, noise, seed + 1)
    out = run_ekf(m, pop, co)
    R = prepare_residuals(out.z, co.mask, co.t, m.burn_in, pop.periods, normalize=False)
    Nobs = int(R.active.sum(1).max())
    alpha = 0.05
    logA = sr_threshold_for_fa(Nobs, alpha)
    s = ShiryaevRoberts().stat(R)
    mx = np.nanmax(np.where(np.isnan(s), -np.inf, s), axis=1)
    fa = float(np.mean(mx >= logA))
    z = R.Z[R.active]
    eps = estimate_epsilon(R.Z[R.active].reshape(1, -1))
    return dict(model=model_name, N_obs=Nobs, alpha_target=alpha, sr_fa_empirical=fa,
                resid_sd=float(np.std(z)), resid_kurtosis=float(np.mean(((z - z.mean()) / z.std()) ** 4)),
                eps_hat=eps, robust_bound=sr_robust_fa_bound(Nobs, np.exp(logA), eps))


def run_all(seed=0):
    return dict(sr_false_alarm=check_sr_false_alarm(seed=seed), sr_delay=check_sr_delay(seed=seed + 1),
                power=check_power(seed=seed + 2), conformal=check_conformal_validity(seed=seed + 3),
                ekf_assumption=[check_ekf_innovations("glucose", seed=seed), check_ekf_innovations("tumor", seed=seed)])
