"""Threshold calibration and evaluation metrics (equal false-alarm rate across detectors)."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

OUT_NONE, OUT_FA, OUT_DET = 0, 1, 2     # miss | false alarm (alarm before onset) | detection


def per_patient_max(stat: np.ndarray, Tm: np.ndarray, t_from=None, t_to=None):
    """Max statistic per patient over observations in [t_from, t_to) (NaN-aware); -inf if none."""
    s = np.where(np.isnan(stat), -np.inf, stat)
    if t_from is not None:
        tf = np.asarray(t_from, float).reshape(-1, 1) if np.ndim(t_from) else t_from
        s = np.where(Tm >= tf, s, -np.inf)
    if t_to is not None:
        tt = np.asarray(t_to, float).reshape(-1, 1) if np.ndim(t_to) else t_to
        s = np.where(Tm < tt, s, -np.inf)
    return s.max(axis=1)


def calibrate_threshold(stat_null: np.ndarray, Tm_null: np.ndarray, alpha: float,
                        t_to: float | None = None) -> float:
    """Threshold h such that a fraction alpha of no-failure patients raise an alarm in the monitored
    window (all monitored observations, or those before t_to). Uses the empirical quantile with the
    finite-sample (n+1) correction."""
    mx = per_patient_max(stat_null, Tm_null, t_to=t_to)
    mx = mx[np.isfinite(mx)]
    if mx.size == 0:
        return np.inf
    n = mx.size
    k = min(int(np.ceil((n + 1) * (1 - alpha))), n)
    return float(np.sort(mx)[k - 1])


def first_alarm(stat: np.ndarray, Tm: np.ndarray, h: float):
    """First time each patient's statistic exceeds h (inf if never)."""
    over = np.where(np.isnan(stat), False, stat > h)
    any_ = over.any(axis=1)
    idx = np.argmax(over, axis=1)
    t = Tm[np.arange(stat.shape[0]), idx]
    return np.where(any_, t, np.inf), np.where(any_, idx, -1)


def evaluate(stat: np.ndarray, Tm: np.ndarray, h: float, onset: np.ndarray, horizon: float):
    """Per-patient outcomes for a statistic and threshold. Returns a dict of arrays."""
    onset = np.asarray(onset, float)
    t_alarm, idx = first_alarm(stat, Tm, h)
    outcome = np.where(~np.isfinite(t_alarm), OUT_NONE, np.where(t_alarm < onset, OUT_FA, OUT_DET))
    delay = np.where(outcome == OUT_DET, t_alarm - onset, np.nan)
    score_post = per_patient_max(stat, Tm, t_from=onset)
    score_pre = per_patient_max(stat, Tm, t_to=onset)
    return dict(t_alarm=t_alarm, alarm_idx=idx, outcome=outcome, delay=delay,
                score_post=score_post, score_pre=score_pre, onset=onset, horizon=horizon)


def summarize(ev: dict) -> dict:
    oc = ev["outcome"]
    n = oc.size
    det = oc == OUT_DET
    fa = oc == OUT_FA
    ok = ~fa                                   # patients that were still being monitored at onset
    cens = ev["horizon"] - ev["onset"]
    d_cens = np.where(det, ev["delay"], cens)
    return dict(
        n=int(n),
        far_pre=float(fa.mean()),
        detect_rate=float(det.mean()),
        detect_rate_nofa=float(det[ok].mean()) if ok.any() else np.nan,
        median_delay=float(np.nanmedian(ev["delay"])) if det.any() else np.nan,
        mean_delay_cens=float(d_cens[ok].mean()) if ok.any() else np.nan,
    )


def auroc(score_fault: np.ndarray, score_null: np.ndarray):
    sf = np.where(np.isfinite(score_fault), score_fault, -1e9)
    sn = np.where(np.isfinite(score_null), score_null, -1e9)
    y = np.r_[np.ones(sf.size), np.zeros(sn.size)]
    s = np.r_[sf, sn]
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, s))


def roc_points(score_fault, score_null):
    sf = np.where(np.isfinite(score_fault), score_fault, -1e9)
    sn = np.where(np.isfinite(score_null), score_null, -1e9)
    y = np.r_[np.ones(sf.size), np.zeros(sn.size)]
    fpr, tpr, _ = roc_curve(y, np.r_[sf, sn])
    return fpr, tpr


def far_full(stat_null, Tm_null, h):
    """False-alarm probability over the whole monitored window for a no-failure cohort."""
    mx = per_patient_max(stat_null, Tm_null)
    return float(np.mean(mx > h))


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot=2000, seed=0, stat=np.mean):
    """Bootstrap CI of stat(a) - stat(b) for paired per-patient outcomes (same patients)."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = a.size
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    n = a.size
    if n == 0:
        return np.nan, np.nan, np.nan
    idx = rng.integers(0, n, (n_boot, n))
    d = np.array([stat(a[i]) - stat(b[i]) for i in idx]) if n < 50 else \
        stat(a[idx], axis=1) - stat(b[idx], axis=1)
    return float(stat(a) - stat(b)), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def expected_calibration_error(proba: np.ndarray, y_true: np.ndarray, classes, bins=10):
    conf = proba.max(1)
    pred = np.asarray(classes)[proba.argmax(1)]
    corr = (pred == y_true).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(corr[m].mean() - conf[m].mean())
    return float(ece)
