"""Observation (sampling) schemes on the simulation grid.

Schemes (intervals are in model time units; see SCHEMES):
  dense      regular, short interval       (glucose: hourly;  tumour: daily)
  moderate   regular                       (glucose: daily;   tumour: every 3 days)
  sparse     regular, long interval        (glucose: weekly;  tumour: weekly)
  realistic  irregular "clinic visits": gamma-distributed gaps, patient-specific visit rate
Optional missingness: iid dropout ('random') or contiguous gaps ('blocks').
"""
from __future__ import annotations
import numpy as np

SCHEMES = {
    "glucose": {"dense": 1.0, "moderate": 24.0, "sparse": 168.0, "realistic": 36.0},
    "tumor":   {"dense": 1.0, "moderate": 3.0,  "sparse": 7.0,   "realistic": 5.0},
}


def make_mask(model, scheme: str, P: int, rng, miss_prob: float = 0.0, miss_kind: str = "random",
              freq_cv: float = 0.0, block_len_frac: float = 0.05):
    """Return boolean mask (P, T+1) of observed grid steps."""
    T = model.n_steps
    dt = model.dt
    base = SCHEMES[model.name][scheme]
    mask = np.zeros((P, T + 1), dtype=bool)
    for i in range(P):
        scale = np.exp(rng.normal(-0.5 * freq_cv ** 2, freq_cv)) if freq_cv > 0 else 1.0
        if scheme == "realistic":
            mean_gap = base * scale * np.exp(rng.normal(-0.125, 0.5))   # patient-specific visit rate
            shape = 1.5
            t = rng.uniform(0, mean_gap)
            steps = []
            while t < model.horizon:
                steps.append(int(round(t / dt)))
                t += max(rng.gamma(shape, mean_gap / shape), 2 * dt)
            steps = np.array(steps, dtype=int)
        else:
            interval = base * scale
            t0 = rng.uniform(0, interval)
            steps = np.round(np.arange(t0, model.horizon, interval) / dt).astype(int)
        steps = steps[(steps >= 0) & (steps <= T)]
        mask[i, np.unique(steps)] = True
    if miss_prob > 0:
        if miss_kind == "random":
            drop = rng.random(mask.shape) < miss_prob
            mask &= ~drop
        elif miss_kind == "blocks":
            L = max(1, int(block_len_frac * T))
            nblk = max(1, int(round(miss_prob / block_len_frac)))
            for i in range(P):
                for _ in range(nblk):
                    s = rng.integers(0, T - L)
                    mask[i, s:s + L] = False
        else:
            raise ValueError(miss_kind)
    return mask
