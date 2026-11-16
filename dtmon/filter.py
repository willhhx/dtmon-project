"""Observer-based residual generation: batched extended Kalman filter (EKF) twin.

For each patient the twin integrates the nominal ODE (calibrated parameters, known inputs, NO failure
terms), and at each observation produces the standardised innovation  z = (y - yhat) / sqrt(S),
S = H P H' + R. Under a correct model with a correct noise model, z ~ N(0,1) and is white.
Failures show up as shifts, variance inflation, autocorrelation or periodicity in z.

mode='ekf'       nonlinear predict + linearised update (state correction)
mode='open_loop' no state update (conventional "twin vs data" comparison); used as a baseline / ablation
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class FilterOutput:
    z: np.ndarray          # (P, T+1) standardised innovations, NaN where not observed
    nu_rel: np.ndarray     # relative raw innovation (y-yhat)/yhat
    corr: np.ndarray       # normalised state-correction magnitude
    infl: np.ndarray       # S / R  (uncertainty inflation vs pure measurement noise)
    mask: np.ndarray
    n_resets: np.ndarray = None   # per-patient number of filter re-initialisations


def _rk4_aug(model, x, Phi, u0, u1, u2, p, dt):
    def d(xx, PP, u):
        f = model.f_twin(xx, u, p)
        J = model.jac_twin(xx, u, p)
        return f, np.einsum("pij,pjk->pik", J, PP)
    f1, g1 = d(x, Phi, u0)
    f2, g2 = d(x + dt / 2 * f1, Phi + dt / 2 * g1, u1)
    f3, g3 = d(x + dt / 2 * f2, Phi + dt / 2 * g2, u1)
    f4, g4 = d(x + dt * f3, Phi + dt * g3, u2)
    xn = x + dt / 6 * (f1 + 2 * f2 + 2 * f3 + f4)
    Pn = Phi + dt / 6 * (g1 + 2 * g2 + 2 * g3 + g4)
    return xn, Pn


def run_ekf(model, pop, cohort, mode: str = "ekf", q_frac: float = 0.03, p0_frac: float = 0.05,
            obs_floor: float | None = None, cv_assumed_scale: float = 1.0) -> FilterOutput:
    """Run the batched twin filter. q_frac: process-noise sd as a fraction of the state scale per
    sqrt(time unit); p0_frac: initial state uncertainty fraction."""
    assert mode in ("ekf", "open_loop")
    P, T1 = cohort.y.shape
    T = T1 - 1
    n, dt = model.n, model.dt
    p = pop.twin_params
    scale = model.state_scale(p)                     # (P, n)
    x = pop.x0_twin.copy()
    x0_init = x.copy()
    Pm = np.zeros((P, n, n))
    Pm[:, np.arange(n), np.arange(n)] = (p0_frac * scale) ** 2
    P0 = Pm.copy()
    n_resets = np.zeros(P, int)
    Qdt = np.zeros((P, n, n))
    Qdt[:, np.arange(n), np.arange(n)] = (q_frac * scale) ** 2 * dt
    o = model.obs_state
    floor = obs_floor if obs_floor is not None else (20.0 if model.name == "glucose" else 1.0)
    z = np.full((P, T1), np.nan, dtype=np.float32)
    nu_rel = np.full((P, T1), np.nan, dtype=np.float32)
    corr = np.full((P, T1), np.nan, dtype=np.float32)
    infl = np.full((P, T1), np.nan, dtype=np.float32)
    mask, y, cv = cohort.mask, cohort.y, cohort.noise_cv
    eye = np.eye(n)[None]
    inp = pop.inp
    for k in range(T1):
        if k > 0:
            u0, u1, u2 = inp[:, :, 2 * (k - 1)], inp[:, :, 2 * (k - 1) + 1], inp[:, :, 2 * k]
            Phi0 = np.broadcast_to(eye, (P, n, n)).copy()
            x, Phi = _rk4_aug(model, x, Phi0, u0, u1, u2, p, dt)
            x = model.clip_state(np.nan_to_num(x, nan=1.0, posinf=1e6, neginf=0.0))
            Pm = np.einsum("pij,pjk,plk->pil", Phi, Pm, Phi) + Qdt
            Pm = np.clip(np.nan_to_num(Pm, nan=0.0, posinf=1e8, neginf=-1e8), -1e8, 1e8)
            bad = ~(np.isfinite(x).all(1) & np.isfinite(Pm).all((1, 2)))
            if bad.any():                                      # safeguard: re-initialise diverged filters
                x[bad] = x0_init[bad]
                Pm[bad] = P0[bad]
                n_resets[bad] += 1
        idx = np.nonzero(mask[:, k])[0]
        if idx.size == 0:
            continue
        yh = x[idx, o]
        yhs = np.maximum(yh, floor)
        R = (cv_assumed_scale * cv[idx] * yhs) ** 2     # cv_assumed_scale != 1 -> misspecified noise model
        Pi = Pm[idx]
        S = np.maximum(Pi[:, o, o], 0.0) + R
        nu = y[idx, k] - yh
        z[idx, k] = nu / np.sqrt(S)
        nu_rel[idx, k] = nu / yhs
        infl[idx, k] = np.minimum(S / R, 1e6)
        K = Pi[:, :, o] / S[:, None]                  # (m, n)
        dx = K * nu[:, None]
        sd_prior = np.sqrt(np.maximum(np.einsum("pii->pi", Pi), (1e-3 * scale[idx]) ** 2))
        corr[idx, k] = np.minimum(np.sqrt(np.sum((dx / sd_prior) ** 2, axis=1)), 1e3)
        if mode == "ekf":
            x[idx] = model.clip_state(x[idx] + dx)
            A = np.broadcast_to(eye, (idx.size, n, n)).copy()
            A[:, :, o] -= K                              # (I - K H)
            Pn = np.einsum("pij,pjk,plk->pil", A, Pi, A) + R[:, None, None] * K[:, :, None] * K[:, None, :]
            Pm[idx] = 0.5 * (Pn + np.transpose(Pn, (0, 2, 1)))     # Joseph form: stays PSD
    return FilterOutput(z, nu_rel, corr, infl, mask, n_resets)


def compress(arr: np.ndarray, mask: np.ndarray, t: np.ndarray):
    """Convert grid-aligned (P, T+1) data to observation-ordered (P, M) arrays (NaN / +inf padded).
    Returns (A, Tm, counts) with A[p, m] the m-th observation of patient p and Tm its time."""
    P = arr.shape[0]
    counts = mask.sum(1)
    M = int(counts.max()) if counts.size else 0
    A = np.full((P, max(M, 1)), np.nan, dtype=np.float32)
    Tm = np.full((P, max(M, 1)), np.inf, dtype=np.float64)
    for i in range(P):
        ii = np.nonzero(mask[i])[0]
        A[i, :ii.size] = arr[i, ii]
        Tm[i, :ii.size] = t[ii]
    return A, Tm, counts
