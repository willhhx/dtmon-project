"""Virtual-patient cohort generation with failure injection and noisy sparse observations."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from .models import (get_model, rk4_truth, FaultSpec, KIND_NONE, KIND_DRIFT, KIND_STRUCT,
                     KIND_FEEDBACK, KIND_NAMES, SUB_NAMES)
from .sampling import make_mask

LEVELS = {"small": 0, "medium": 1, "large": 2}


@dataclass
class PatientPop:
    """Patients: parameters, initial states, known inputs. Independent of failure/sampling."""
    model: object
    P: int
    params: dict
    twin_params: dict
    S0: np.ndarray
    x0_twin: np.ndarray
    inp: np.ndarray
    treated: np.ndarray
    periods: np.ndarray


def make_population(model, P: int, seed: int, cv_scale: float = 1.0, cal_sigma: float = 0.05,
                    p0_frac: float = 0.05, treat_frac: float = 0.5):
    """Sample P virtual patients. Everything is a deterministic function of `seed` so that
    different failure/sampling scenarios share the SAME patients (common random numbers)."""
    rng = np.random.default_rng(seed)
    params = model.sample_params(rng, P, cv_scale)
    S0 = model.truth_init(params)
    t_half = np.arange(2 * model.n_steps + 1) * model.dt / 2.0
    inp, treated = model.build_inputs(rng, P, t_half, treat_frac)
    twin = {k: v * np.exp(rng.normal(0, cal_sigma, P)) for k, v in params.items()}
    if "Gb" in twin:                  # keep h == Gb in the twin
        twin["h"] = twin["Gb"].copy()
    x0_twin = model.initial_state(twin) * (1 + p0_frac * rng.normal(0, 1, (P, model.n)))
    x0_twin = model.clip_state(x0_twin)
    periods = model.candidate_period * rng.uniform(0.85, 1.15, P)
    return PatientPop(model, P, params, twin, S0, x0_twin, inp, treated, periods)


def make_fault(model, pop: PatientPop, kind: int, level="medium", sub="random", seed: int = 0,
               onset=None) -> FaultSpec:
    """Build the per-patient fault description. `level`: 'small'|'medium'|'large'|float|'mixed'."""
    P = pop.P
    rng = np.random.default_rng(seed + 991 * (kind + 1))
    if kind == KIND_NONE:
        return FaultSpec.none(P)
    if isinstance(level, str) and level == "mixed":
        mags = np.array(model.mag_table[kind])[rng.integers(0, 3, P)]
    elif isinstance(level, str):
        mags = np.full(P, model.mag_table[kind][LEVELS[level]])
    else:
        mags = np.full(P, float(level))
    if kind == KIND_STRUCT:
        subs = (rng.integers(0, 4, P) if sub == "random" else np.full(P, int(sub)))
    else:
        subs = np.zeros(P, int)
    t_on = model.t_change if onset is None else onset
    onsets = np.full(P, float(t_on)) + rng.uniform(-0.05, 0.05, P) * model.horizon * 0.1
    ramp = np.full(P, model.ramp_default)
    if kind == KIND_STRUCT:
        ramp = np.full(P, model.ramp_default * 0.25)   # structural changes switch in faster
    return FaultSpec(np.full(P, kind, int), subs.astype(int), mags, onsets, ramp, pop.periods.copy())


def simulate_truth(pop: PatientPop, fault: FaultSpec):
    """Noise-free observed-state trajectory (P, T+1) under the given fault."""
    y, _ = rk4_truth(pop.model, pop.S0, pop.inp, pop.params, fault)
    return y


@dataclass
class Cohort:
    model_name: str
    t: np.ndarray
    y: np.ndarray            # observed (NaN where not observed)
    y_clean: np.ndarray      # noise-free observed state on the grid
    mask: np.ndarray
    noise_cv: np.ndarray
    fault: FaultSpec
    scheme: str
    meta: dict = field(default_factory=dict)

    @property
    def P(self):
        return self.y.shape[0]


def observe(model, pop: PatientPop, y_clean, fault: FaultSpec, scheme: str, noise_cv, seed: int,
            miss_prob: float = 0.0, miss_kind: str = "random", freq_cv: float = 0.0) -> Cohort:
    """Apply sampling mask and multiplicative measurement noise."""
    rng = np.random.default_rng(seed)
    P = pop.P
    mask = make_mask(model, scheme, P, rng, miss_prob, miss_kind, freq_cv)
    cv = np.full(P, float(noise_cv)) if np.isscalar(noise_cv) else np.asarray(noise_cv, float)
    eps = rng.normal(0, 1, y_clean.shape).astype(np.float32)
    y = y_clean * (1.0 + cv[:, None].astype(np.float32) * eps)
    y = np.maximum(y, 1e-3)
    y = np.where(mask, y, np.nan).astype(np.float32)
    return Cohort(model.name, model.time_grid(), y, y_clean, mask, cv, fault, scheme,
                  dict(miss_prob=miss_prob, miss_kind=miss_kind, freq_cv=freq_cv))
