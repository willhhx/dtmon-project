"""
Two published ODE systems used as digital-twin benchmarks, with batched (vectorised over
patients) right-hand sides, analytic Jacobians for the twin, and a failure-injection layer
for the "truth" simulator.

Model A  glucose-insulin: Bergman minimal model (Bergman et al. 1979, 1981) extended with a
         pancreatic insulin-secretion feedback term (glucose-dependent secretion).
Model B  tumour-immune: Kuznetsov, Makalkin, Taylor & Perelson (1994) model, in the
         non-dimensional form commonly used in the literature, plus a cytotoxic treatment term.

IMPORTANT: parameter values are representative literature values, jittered across virtual
patients. They are NOT clinically validated. Calibrate/replace before making biological claims.

Failure codes   kind: 0 none | 1 parameter drift | 2 structural error | 3 feedback loss
Structure subtypes: 0 hidden periodic forcing | 1 nonlinear saturation
                    2 delayed interaction     | 3 omitted dynamics (hidden state)
The twin NEVER contains any failure term.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

KIND_NONE, KIND_DRIFT, KIND_STRUCT, KIND_FEEDBACK = 0, 1, 2, 3
SUB_PERIODIC, SUB_SATURATION, SUB_DELAY, SUB_OMITTED = 0, 1, 2, 3
KIND_NAMES = {0: "none", 1: "drift", 2: "structure", 3: "feedback"}
SUB_NAMES = {0: "periodic", 1: "saturation", 2: "delay", 3: "omitted"}


@dataclass
class FaultSpec:
    """Per-patient failure description (all arrays have shape (P,))."""
    kind: np.ndarray
    sub: np.ndarray
    mag: np.ndarray       # effective magnitude (model/fault specific scale)
    onset: np.ndarray     # change time (model time units); +inf if no fault
    ramp: np.ndarray      # ramp length (model time units), >0
    period: np.ndarray    # period of hidden forcing (model time units)

    def phase(self, t: float) -> np.ndarray:
        a = np.clip((t - self.onset) / self.ramp, 0.0, 1.0)
        a = np.where(np.isfinite(self.onset), a, 0.0)
        return a

    @staticmethod
    def none(P: int) -> "FaultSpec":
        return FaultSpec(np.zeros(P, int), np.zeros(P, int), np.zeros(P),
                         np.full(P, np.inf), np.ones(P), np.ones(P))


def _lognormal(rng, nominal, cv, P):
    s = np.sqrt(np.log(1 + cv ** 2))
    return nominal * rng.lognormal(-0.5 * s * s, s, P)


# ---------------------------------------------------------------------------------------
class BaseModel:
    name = "base"
    n = 0                   # twin state dimension
    nh = 0                  # hidden truth-only states
    time_unit = "h"
    dt = 0.1                # simulation / filter grid step
    horizon = 100.0
    t_change = 50.0
    burn_in = 10.0
    n_inputs = 1
    obs_state = 0
    candidate_period = 24.0
    ramp_default = 1.0
    # failure magnitude tables (small, medium, large)
    mag_table = {1: (0.2, 0.4, 0.6), 2: (0.3, 0.6, 1.0), 3: (0.4, 0.7, 1.0)}
    sampling = {}           # filled by subclasses (see sampling.py)

    # to be provided by subclasses
    def sample_params(self, rng, P, cv_scale=1.0): raise NotImplementedError
    def initial_state(self, params): raise NotImplementedError
    def build_inputs(self, rng, P, t_half, treat_frac): raise NotImplementedError
    def f_twin(self, x, u, p): raise NotImplementedError
    def jac_twin(self, x, u, p): raise NotImplementedError
    def f_truth(self, t, S, u, p, f): raise NotImplementedError
    def h(self, x): return x[:, self.obs_state]

    def H(self, x):
        Hm = np.zeros((x.shape[0], self.n))
        Hm[:, self.obs_state] = 1.0
        return Hm

    def state_scale(self, p):
        """Typical magnitude of each state (P, n), used for process-noise scaling."""
        raise NotImplementedError

    def clip_state(self, x):
        return np.maximum(x, 1e-6)

    @property
    def n_steps(self):
        return int(round(self.horizon / self.dt))

    def time_grid(self):
        return np.arange(self.n_steps + 1) * self.dt


def rk4_truth(model: BaseModel, S0, inp, p, f: FaultSpec):
    """Vectorised RK4 for the truth system. inp has shape (P, k, 2T+1) on the half-grid.
    Returns the observed-state trajectory (P, T+1) and the final states."""
    P = S0.shape[0]
    T = model.n_steps
    dt = model.dt
    S = S0.copy()
    out = np.empty((P, T + 1), dtype=np.float32)
    out[:, 0] = model.h(S[:, :model.n])
    for k in range(T):
        t = k * dt
        u0, u1, u2 = inp[:, :, 2 * k], inp[:, :, 2 * k + 1], inp[:, :, 2 * k + 2]
        k1 = model.f_truth(t, S, u0, p, f)
        k2 = model.f_truth(t + dt / 2, S + dt / 2 * k1, u1, p, f)
        k3 = model.f_truth(t + dt / 2, S + dt / 2 * k2, u1, p, f)
        k4 = model.f_truth(t + dt, S + dt * k3, u2, p, f)
        S = S + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        S[:, :model.n] = model.clip_state(S[:, :model.n])
        out[:, k + 1] = model.h(S[:, :model.n])
    return out, S


# ---------------------------------------------------------------------------------------
class GlucoseInsulin(BaseModel):
    """Bergman minimal model + glucose-dependent insulin secretion (feedback loop).
        G' = -p1 (G-Gb) - X G + D(t)
        X' = -p2 X + p3 (I-Ib)
        I' = -n (I-Ib) + gamma max(G-h,0) + U(t)
    time unit: hours; G mg/dL; I uU/mL.  Observation: G."""
    name = "glucose"
    n, nh = 3, 2
    time_unit = "h"
    dt = 0.125
    horizon = 28 * 24.0
    t_change = 14 * 24.0
    burn_in = 3 * 24.0
    n_inputs = 2             # meal D(t), insulin bolus U(t)
    candidate_period = 24.0
    ramp_default = 24.0
    obs_state = 0
    mag_table = {1: (0.2, 0.4, 0.6), 2: (0.3, 0.6, 1.0), 3: (0.4, 0.7, 1.0)}
    PERIODIC_AMP = 25.0      # mg/dL/h at magnitude 1
    OMIT_GAIN = 30.0
    SAT_GAIN = 6.0

    def sample_params(self, rng, P, cv_scale=1.0):
        c = lambda x, cv: _lognormal(rng, x, cv * cv_scale, P)
        p = dict(
            p1=c(0.6, 0.20), p2=c(1.0, 0.20), p3=c(0.015, 0.25),
            n=c(4.0, 0.15), gamma=c(1.3, 0.25),
            Gb=c(90.0, 0.08), Ib=c(10.0, 0.20),
        )
        p["h"] = p["Gb"].copy()
        return p

    def initial_state(self, p):
        return np.stack([p["Gb"], np.zeros_like(p["Gb"]), p["Ib"]], axis=1)

    def state_scale(self, p):
        return np.stack([p["Gb"] * 0.5, np.full_like(p["Gb"], 0.2), p["Ib"] * 2.0], axis=1)

    def build_inputs(self, rng, P, t_half, treat_frac):
        """Known inputs: meals (3/day, jittered) and insulin boluses for treated patients."""
        days = int(np.ceil(t_half[-1] / 24.0)) + 1
        D = np.zeros((P, len(t_half)), dtype=np.float32)
        U = np.zeros((P, len(t_half)), dtype=np.float32)
        treated = rng.random(P) < treat_frac
        dose_ratio = np.where(treated, rng.uniform(0.02, 0.06, P), 0.0)  # uU/mL per gram carb
        for d in range(days):
            for hour, carb in ((7.0, 55.0), (12.5, 70.0), (19.0, 80.0)):
                tm = d * 24.0 + hour + rng.normal(0, 0.5, P)
                c = carb * rng.lognormal(0, 0.25, P)
                sd = 0.6
                x = t_half[None, :] - (tm[:, None] + 0.5)
                pulse = np.exp(-0.5 * (x / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
                D += (c[:, None] * 1.4 * pulse).astype(np.float32)           # mg/dL per gram
                xb = t_half[None, :] - (tm[:, None] - 0.25)
                pb = np.exp(-0.5 * (xb / 0.4) ** 2) / (0.4 * np.sqrt(2 * np.pi))
                U += ((c * dose_ratio)[:, None] * pb).astype(np.float32)
        return np.stack([D, U], axis=1), treated

    def f_twin(self, x, u, p):
        G, X, I = x[:, 0], x[:, 1], x[:, 2]
        dG = -p["p1"] * (G - p["Gb"]) - X * G + u[:, 0]
        dX = -p["p2"] * X + p["p3"] * (I - p["Ib"])
        dI = -p["n"] * (I - p["Ib"]) + p["gamma"] * np.maximum(G - p["h"], 0.0) + u[:, 1]
        return np.stack([dG, dX, dI], axis=1)

    def jac_twin(self, x, u, p):
        G, X = x[:, 0], x[:, 1]
        J = np.zeros((x.shape[0], 3, 3))
        J[:, 0, 0] = -p["p1"] - X
        J[:, 0, 1] = -G
        J[:, 1, 1] = -p["p2"]
        J[:, 1, 2] = p["p3"]
        J[:, 2, 0] = p["gamma"] * (G > p["h"])
        J[:, 2, 2] = -p["n"]
        return J

    def f_truth(self, t, S, u, p, f):
        G, X, I, h1, h2 = S.T
        a = f.phase(t)
        aD = a * (f.kind == KIND_DRIFT)
        aS = a * (f.kind == KIND_STRUCT)
        aF = a * (f.kind == KIND_FEEDBACK)
        p3 = p["p3"] * (1.0 - f.mag * aD)                       # insulin sensitivity loss
        gam = p["gamma"] * (1.0 - f.mag * aF)                   # secretion feedback loss
        isS = (f.kind == KIND_STRUCT)
        sP = isS & (f.sub == SUB_PERIODIC)
        sSat = isS & (f.sub == SUB_SATURATION)
        sDel = isS & (f.sub == SUB_DELAY)
        sOm = isS & (f.sub == SUB_OMITTED)
        onset = np.where(np.isfinite(f.onset), f.onset, 0.0)
        Xeff = X
        Xeff = np.where(sDel, (1 - aS) * X + aS * h1, Xeff)     # delayed insulin action
        Xeff = np.where(sSat, X / (1.0 + self.SAT_GAIN * f.mag * aS * X), Xeff)  # saturation
        dG = -p["p1"] * (G - p["Gb"]) - Xeff * G + u[:, 0]
        dG = dG + np.where(sP, aS * f.mag * self.PERIODIC_AMP *
                           np.sin(2 * np.pi * (t - onset) / f.period), 0.0)
        dG = dG - np.where(sOm, aS * f.mag * self.OMIT_GAIN * h2, 0.0)
        dX = -p["p2"] * X + p3 * (I - p["Ib"])
        dI = -p["n"] * (I - p["Ib"]) + gam * np.maximum(G - p["h"], 0.0) + u[:, 1]
        tau = 0.5 + 6.0 * np.maximum(f.mag, 0.05)
        dh1 = (X - h1) / tau
        dh2 = -0.5 * h2 + 0.05 * (G - p["Gb"])
        return np.stack([dG, dX, dI, dh1, dh2], axis=1)

    def truth_init(self, p):
        x0 = self.initial_state(p)
        return np.concatenate([x0, np.zeros((x0.shape[0], 2))], axis=1)

    def clip_state(self, x):
        x = x.copy()
        x[:, 0] = np.clip(x[:, 0], 20.0, 1000.0)
        x[:, 1] = np.clip(x[:, 1], 0.0, 50.0)
        x[:, 2] = np.clip(x[:, 2], 0.0, 1000.0)
        return x


# ---------------------------------------------------------------------------------------
class TumorImmune(BaseModel):
    """Kuznetsov-Taylor tumour-immune model (non-dimensional; time unit: days).
        x' = sigma + rho x y/(eta+y) - mu x y - delta x                (effector cells)
        y' = alpha y (1 - beta y) - x y - kT u(t) y                     (tumour; u = therapy)
    Observation: tumour burden y (multiplicative noise)."""
    name = "tumor"
    n, nh = 2, 2
    time_unit = "d"
    dt = 0.1
    horizon = 240.0
    t_change = 120.0
    burn_in = 40.0
    n_inputs = 1
    candidate_period = 14.0
    ramp_default = 14.0
    obs_state = 1
    mag_table = {1: (0.15, 0.30, 0.50), 2: (0.3, 0.6, 1.0), 3: (0.04, 0.08, 0.15)}
    PERIODIC_AMP = 0.5
    SAT_GAIN = 0.02
    OMIT_GAIN = 0.4

    def sample_params(self, rng, P, cv_scale=1.0):
        c = lambda x, cv: _lognormal(rng, x, cv * cv_scale, P)
        return dict(
            sigma=c(0.1181, 0.10), rho=c(1.131, 0.10), eta=c(20.19, 0.10),
            mu=c(0.00311, 0.10), delta=c(0.3743, 0.10), alpha=c(1.636, 0.10),
            beta=c(0.002, 0.10), kT=c(0.8, 0.20),
        )

    def initial_state(self, p):
        P = p["sigma"].shape[0]
        return np.stack([p["sigma"] / p["delta"], np.full(P, 5.0)], axis=1)

    def state_scale(self, p):
        return np.stack([np.full_like(p["sigma"], 1.0), np.full_like(p["sigma"], 50.0)], axis=1)

    def build_inputs(self, rng, P, t_half, treat_frac):
        """Known therapy: 3-day cycles every 21 days for treated patients (smooth pulses)."""
        treated = rng.random(P) < treat_frac
        U = np.zeros((P, len(t_half)), dtype=np.float32)
        start = rng.uniform(5.0, 15.0, P)
        cyc = 21.0
        dur = 3.0
        s = 0.5
        ncyc = int(t_half[-1] // cyc) + 2
        for c_ in range(ncyc):
            t0 = start + c_ * cyc
            a = 1.0 / (1 + np.exp(-(t_half[None, :] - t0[:, None]) / s))
            b = 1.0 / (1 + np.exp(-(t_half[None, :] - (t0[:, None] + dur)) / s))
            U += ((a - b) * treated[:, None]).astype(np.float32)
        return U[:, None, :], treated

    def f_twin(self, x, u, p):
        xe, y = x[:, 0], x[:, 1]
        dx = p["sigma"] + p["rho"] * xe * y / (p["eta"] + y) - p["mu"] * xe * y - p["delta"] * xe
        dy = p["alpha"] * y * (1 - p["beta"] * y) - xe * y - p["kT"] * u[:, 0] * y
        return np.stack([dx, dy], axis=1)

    def jac_twin(self, x, u, p):
        xe, y = x[:, 0], x[:, 1]
        J = np.zeros((x.shape[0], 2, 2))
        J[:, 0, 0] = p["rho"] * y / (p["eta"] + y) - p["mu"] * y - p["delta"]
        J[:, 0, 1] = p["rho"] * xe * p["eta"] / (p["eta"] + y) ** 2 - p["mu"] * xe
        J[:, 1, 0] = -y
        J[:, 1, 1] = p["alpha"] * (1 - 2 * p["beta"] * y) - xe - p["kT"] * u[:, 0]
        return J

    def f_truth(self, t, S, u, p, f):
        xe, y, h1, h2 = S.T
        a = f.phase(t)
        aD = a * (f.kind == KIND_DRIFT)
        aS = a * (f.kind == KIND_STRUCT)
        aF = a * (f.kind == KIND_FEEDBACK)
        alpha = p["alpha"] * (1.0 + f.mag * aD)                 # tumour growth-rate increase
        rho = p["rho"] * (1.0 - f.mag * aF)                     # immune recruitment loss (suppression)
        isS = (f.kind == KIND_STRUCT)
        sP = isS & (f.sub == SUB_PERIODIC)
        sSat = isS & (f.sub == SUB_SATURATION)
        sDel = isS & (f.sub == SUB_DELAY)
        sOm = isS & (f.sub == SUB_OMITTED)
        onset = np.where(np.isfinite(f.onset), f.onset, 0.0)
        x_kill = xe
        x_kill = np.where(sDel, (1 - aS) * xe + aS * h1, x_kill)           # delayed immune effect
        kill = x_kill * y
        kill = np.where(sSat, x_kill * y / (1.0 + self.SAT_GAIN * f.mag * aS * y), kill)  # saturation
        kill = np.where(sOm, kill * (1.0 - aS * f.mag * 0.8 * h2 / (1.0 + h2)), kill)    # suppressor
        grow = alpha * y * (1 - p["beta"] * y)
        grow = grow * np.where(sP, 1.0 + aS * f.mag * self.PERIODIC_AMP *
                               np.sin(2 * np.pi * (t - onset) / f.period), 1.0)
        dx = p["sigma"] + rho * xe * y / (p["eta"] + y) - p["mu"] * xe * y - p["delta"] * xe
        dy = grow - kill - p["kT"] * u[:, 0] * y
        tau = 1.0 + 8.0 * np.maximum(f.mag, 0.05)
        dh1 = (xe - h1) / tau
        dh2 = -0.1 * h2 + 0.02 * y
        return np.stack([dx, dy, dh1, dh2], axis=1)

    def truth_init(self, p):
        x0 = self.initial_state(p)
        return np.concatenate([x0, np.stack([x0[:, 0], np.zeros(x0.shape[0])], axis=1)], axis=1)

    def clip_state(self, x):
        x = np.maximum(x, 1e-6)
        x[:, 0] = np.minimum(x[:, 0], 100.0)       # effector cells
        x[:, 1] = np.minimum(x[:, 1], 1500.0)      # tumour burden (carrying capacity ~ 1/beta = 500)
        return x


MODELS = {"glucose": GlucoseInsulin, "tumor": TumorImmune}


def get_model(name: str) -> BaseModel:
    return MODELS[name]()
