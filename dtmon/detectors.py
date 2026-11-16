"""Sequential detectors D1-D10 on observation-ordered standardised residuals.

All detectors work on a `Residuals` object (patients x observations, irregular times) and return a
statistic array (P, M): larger = more evidence of failure; NaN where an observation is padding or
before monitoring starts (t < burn_in). An alarm is the first monitored observation whose statistic
exceeds a threshold h. Thresholds are calibrated on an independent no-failure cohort (see calibrate.py)
so that every detector has the same empirical false-alarm probability.

D1 threshold | D2 moving average | D3 EWMA | D4 CUSUM (mean) | D5 windowed GLR (mean+variance)
D6 Shiryaev-Roberts (mixture) | D7 Bayesian online change-point (Normal-Gamma)
D8 conformal residual alarm | D9 multi-scale residual detector | D10 hybrid ensemble
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from scipy.special import gammaln, logsumexp

EPS = 1e-12


# ----------------------------------------------------------------------------------------------
@dataclass
class Residuals:
    Z: np.ndarray            # (P, M) standardised residuals (NaN = padding)
    Tm: np.ndarray           # (P, M) observation times (inf = padding)
    active: np.ndarray       # (P, M) bool: observation is within the monitoring window
    cal: np.ndarray          # (P, B) |z| of burn-in observations (NaN padded), for conformal
    burn_in: float
    period: np.ndarray       # (P,) candidate period of hidden periodic forcing (per patient scale)
    aux: dict = field(default_factory=dict)   # extra per-observation arrays (corr, infl, nu_rel)
    cache: dict = field(default_factory=dict) # memoised detector statistics (by detector name)

    @property
    def P(self):
        return self.Z.shape[0]

    @property
    def M(self):
        return self.Z.shape[1]


def prepare_residuals(z_grid, mask, t_grid, burn_in, period, normalize=True, clip=8.0,
                      n0=10.0, aux_grid: dict | None = None) -> Residuals:
    """Compress grid residuals to observation order, optionally apply per-patient 'uncertainty
    normalisation' estimated on the burn-in window (centering + variance, both shrunk towards 0 / 1),
    and winsorise to +-clip. Burn-in observations are used only as reference data."""
    P = z_grid.shape[0]
    counts = mask.sum(1)
    M = max(int(counts.max()), 1)
    Z = np.full((P, M), np.nan)
    Tm = np.full((P, M), np.inf)
    aux = {k: np.full((P, M), np.nan) for k in (aux_grid or {})}
    for i in range(P):
        ii = np.nonzero(mask[i])[0]
        Z[i, :ii.size] = z_grid[i, ii]
        Tm[i, :ii.size] = t_grid[ii]
        for k, a in (aux_grid or {}).items():
            aux[k][i, :ii.size] = a[i, ii]
    valid = np.isfinite(Z)
    burn = valid & (Tm < burn_in)
    active = valid & (Tm >= burn_in)
    if normalize:
        n = burn.sum(1).astype(float)
        zb = np.where(burn, Z, 0.0)
        mean_b = zb.sum(1) / np.maximum(n, 1)
        var_b = np.where(burn, (Z - mean_b[:, None]) ** 2, 0.0).sum(1) / np.maximum(n - 1, 1)
        c = n / (n + n0) * mean_b
        s = np.sqrt((np.maximum(n - 1, 0) * var_b + n0 * 1.0) / (np.maximum(n - 1, 0) + n0))
        Z = (Z - c[:, None]) / s[:, None]
    Z = np.clip(Z, -clip, clip)
    B = max(int(burn.sum(1).max()), 1)
    cal = np.full((P, B), np.nan)
    for i in range(P):
        v = np.abs(Z[i][burn[i]])
        cal[i, :v.size] = v
    return Residuals(Z, Tm, active, cal, burn_in, np.asarray(period, float), aux)


# ----------------------------------------------------------------------------------------------
def _csum(x, active):
    """Cumulative sum along axis 1 of x where active (others 0), prefixed by a zero column."""
    c = np.cumsum(np.where(active, x, 0.0), axis=1)
    return np.concatenate([np.zeros((x.shape[0], 1)), c], axis=1)


def _window_sum(C, w):
    """Sum over the last w columns for every position: out[:, m] = C[:, m+1] - C[:, max(m+1-w, 0)]."""
    M = C.shape[1] - 1
    idx = np.maximum(np.arange(1, M + 1) - w, 0)
    return C[:, 1:] - C[:, idx]


def _count_window(active, w):
    return _window_sum(_csum(np.ones(active.shape), active), w)


class Detector:
    name = "base"

    def fit(self, R: Residuals):  # calibration data from a no-failure cohort (optional)
        return self

    def stat(self, R: Residuals) -> np.ndarray:
        raise NotImplementedError

    def stat_cached(self, R: Residuals) -> np.ndarray:
        if self.name not in R.cache:
            R.cache[self.name] = self.stat(R)
        return R.cache[self.name]

    def _finish(self, s, R):
        s = np.asarray(s, float)
        return np.where(R.active, s, np.nan)


class Threshold(Detector):
    """D1: instantaneous |z|."""
    name = "D1_threshold"

    def stat(self, R):
        return self._finish(np.abs(R.Z), R)


class MovingAverage(Detector):
    """D2: |moving average of z over the last w observations| * sqrt(count)."""
    name = "D2_movavg"

    def __init__(self, w=6):
        self.w = w

    def stat(self, R):
        C = _csum(R.Z, R.active)
        n = _count_window(R.active, self.w)
        s = np.abs(_window_sum(C, self.w)) / np.sqrt(np.maximum(n, 1))
        return self._finish(s, R)


class EWMA(Detector):
    """D3: exponentially weighted moving average of z, standardised."""
    name = "D3_ewma"

    def __init__(self, lam=0.2):
        self.lam = lam

    def stat(self, R):
        P, M = R.Z.shape
        e = np.zeros(P)
        cnt = np.zeros(P)
        out = np.full((P, M), np.nan)
        lam = self.lam
        for m in range(M):
            a = R.active[:, m]
            z = R.Z[:, m]
            e = np.where(a, lam * z + (1 - lam) * e, e)
            cnt = np.where(a, cnt + 1, cnt)
            var = lam / (2 - lam) * (1 - (1 - lam) ** (2 * np.maximum(cnt, 1)))
            out[:, m] = np.where(a, np.abs(e) / np.sqrt(var), np.nan)
        return out


class CUSUM(Detector):
    """D4: two-sided CUSUM for a mean shift (reference value k)."""
    name = "D4_cusum"

    def __init__(self, k=0.5):
        self.k = k

    def stat(self, R):
        P, M = R.Z.shape
        sp = np.zeros(P)
        sn = np.zeros(P)
        out = np.full((P, M), np.nan)
        for m in range(M):
            a = R.active[:, m]
            z = R.Z[:, m]
            sp = np.where(a, np.maximum(0.0, sp + z - self.k), sp)
            sn = np.where(a, np.maximum(0.0, sn - z - self.k), sn)
            out[:, m] = np.where(a, np.maximum(sp, sn), np.nan)
        return out


class GLR(Detector):
    """D5: windowed generalized likelihood ratio for a simultaneous mean and variance change of an
    N(0,1) sequence, maximised over window lengths l in [lmin, W]."""
    name = "D5_glr"

    def __init__(self, W=30, lmin=5, vfloor=0.2):
        self.W, self.lmin, self.vfloor = W, lmin, vfloor

    def stat(self, R):
        P, M = R.Z.shape
        C1 = _csum(R.Z, R.active)
        C2 = _csum(R.Z ** 2, R.active)
        Cn = _csum(np.ones((P, M)), R.active)
        best = np.full((P, M), -np.inf)
        for l in range(self.lmin, self.W + 1):
            n = Cn[:, l:] - Cn[:, :-l] if M >= l else None
            if n is None:
                break
            s1 = C1[:, l:] - C1[:, :-l]
            s2 = C2[:, l:] - C2[:, :-l]
            ok = (n >= l - 0.5)
            mh = s1 / l
            vh = np.maximum(s2 / l - mh ** 2, self.vfloor)
            llr = 0.5 * l * (vh + mh ** 2 - 1.0 - np.log(vh))
            seg = best[:, l - 1:]
            best[:, l - 1:] = np.where(ok, np.maximum(seg, llr), seg)
        best = np.where(np.isinf(best), 0.0, best)
        return self._finish(best, R)


class ShiryaevRoberts(Detector):
    """D6: mixture Shiryaev-Roberts. One SR recursion per alternative j (mean shifts and variance
    inflations): R^(j)_n = (1 + R^(j)_{n-1}) L_j(z_n); the statistic is log of the AVERAGE
    R_n = (1/K) sum_j R^(j)_n.  Under iid N(0,1) residuals each R^(j)_n - n is a martingale, hence so is
    R_n - n and P_0(alarm within N observations) <= N / A for threshold log A (docs/theory.md, T1).
    R_n >= R^(j)_n / K, so the mixture alarms no later than SR_j run with threshold K*A (T3)."""
    name = "D6_sr"

    def __init__(self, shifts=(-1.5, -0.75, 0.75, 1.5), sds=(1.5, 2.5)):
        self.shifts, self.sds = np.array(shifts), np.array(sds)

    @property
    def K(self):
        return len(self.shifts) + len(self.sds)

    def log_lambdas(self, z):
        """(K, ...) array of log likelihood ratios of each alternative against N(0,1)."""
        comps = [th * z - 0.5 * th ** 2 for th in self.shifts]
        comps += [-np.log(s) - 0.5 * z ** 2 * (1.0 / s ** 2 - 1.0) for s in self.sds]
        return np.stack(comps, axis=0)

    def stat(self, R):
        P, M = R.Z.shape
        K = self.K
        logR = np.full((K, P), -np.inf)
        out = np.full((P, M), np.nan)
        ll = self.log_lambdas(np.nan_to_num(R.Z))            # (K, P, M)
        for m in range(M):
            a = R.active[:, m]
            new = np.logaddexp(0.0, logR) + ll[:, :, m]
            logR = np.where(a[None, :], new, logR)
            out[:, m] = np.where(a, logsumexp(logR, axis=0) - np.log(K), np.nan)
        return out


class BOCPD(Detector):
    """D7: Bayesian online change-point detection (Adams & MacKay 2007) with a Normal-Gamma
    conjugate model and Student-t predictive. Statistic: logit of the posterior probability that the
    current run started within the last `recent_eff` observations, where recent_eff = clip(n_seen/3, 2, recent)
    (adaptive window, see the comment in stat()). Burn-in observations are fed to the filter (to establish
    run lengths) but not scored."""
    name = "D7_bocpd"

    def __init__(self, hazard=1 / 100.0, rmax=50, recent=20, mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0):
        self.hazard, self.rmax, self.recent = hazard, rmax, recent
        self.mu0, self.kappa0, self.alpha0, self.beta0 = mu0, kappa0, alpha0, beta0

    def stat(self, R):
        P, M = R.Z.shape
        Rm = self.rmax
        h = self.hazard
        logp = np.full((P, Rm + 1), -np.inf)
        logp[:, 0] = 0.0
        mu = np.full((P, Rm + 1), self.mu0)
        ka = np.full((P, Rm + 1), self.kappa0)
        al = np.full((P, Rm + 1), self.alpha0)
        be = np.full((P, Rm + 1), self.beta0)
        out = np.full((P, M), np.nan)
        valid = np.isfinite(R.Z)
        Z = np.nan_to_num(R.Z)
        n_seen = np.zeros(P)                                     # observations fed so far (incl. burn-in)
        rl = np.arange(Rm + 1)[None, :]
        for m in range(M):
            v = valid[:, m]
            if not v.any():
                continue
            n_seen = n_seen + v
            x = Z[:, m][:, None]
            # Student-t predictive log pdf
            nu = 2 * al
            scale2 = be * (ka + 1) / (al * ka)
            lp = (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * np.pi * scale2)
                  - (nu + 1) / 2 * np.log1p((x - mu) ** 2 / (nu * scale2)))
            lg = logp + lp                                       # joint with data, per run length
            # growth: r -> r+1 (last bin absorbs)
            grow = lg + np.log1p(-h)
            cp = logsumexp(lg, axis=1) + np.log(h)               # change point -> r = 0
            newlogp = np.full_like(logp, -np.inf)
            newlogp[:, 0] = cp
            newlogp[:, 1:] = grow[:, :-1]
            newlogp[:, -1] = np.logaddexp(newlogp[:, -1], grow[:, -1])
            norm = logsumexp(newlogp, axis=1, keepdims=True)
            newlogp = newlogp - norm
            # parameter update
            mu_n = (ka * mu + x) / (ka + 1)
            ka_n = ka + 1
            al_n = al + 0.5
            be_n = be + ka * (x - mu) ** 2 / (2 * (ka + 1))
            nm = np.empty_like(mu); nk = np.empty_like(ka); na = np.empty_like(al); nb = np.empty_like(be)
            nm[:, 0], nk[:, 0], na[:, 0], nb[:, 0] = self.mu0, self.kappa0, self.alpha0, self.beta0
            nm[:, 1:], nk[:, 1:], na[:, 1:], nb[:, 1:] = mu_n[:, :-1], ka_n[:, :-1], al_n[:, :-1], be_n[:, :-1]
            # last bin merges: keep the most-informed params
            nm[:, -1], nk[:, -1], na[:, -1], nb[:, -1] = mu_n[:, -1], ka_n[:, -1], al_n[:, -1], be_n[:, -1]
            vv = v[:, None]
            logp = np.where(vv, newlogp, logp)
            mu = np.where(vv, nm, mu); ka = np.where(vv, nk, ka)
            al = np.where(vv, na, al); be = np.where(vv, nb, be)
            # adaptive window (bug fix): with a fixed window of `recent` run lengths and fewer than `recent`
            # observations the posterior mass in the window is always 1, the statistic sits at its ceiling for
            # every patient, the calibrated threshold equals the ceiling and the detector can never alarm.
            # The window therefore scales with the number of observations seen: recent_eff = n_seen/3 in [2, recent].
            rec = np.clip(np.floor(n_seen / 3.0), 2, self.recent)[:, None]
            pr = np.exp(logsumexp(np.where(rl < rec, logp, -np.inf), axis=1))
            pr = np.clip(pr, 1e-9, 1 - 1e-9)
            out[:, m] = np.log(pr / (1 - pr))
        return self._finish(out, R)


class ConformalAlarm(Detector):
    """D8: per-patient split-conformal p-values from the burn-in |z| (calibration scores),
    p_t = (1 + #{cal >= |z_t|}) / (n_cal + 1); alarm statistic = sum of -log p over the last w monitored
    observations (Fisher-type combination). Each single p_t is marginally super-uniform under
    exchangeability of burn-in and monitored residuals (assumption).

    Tie handling (bug fix). With few burn-in scores (e.g. 3 daily glucose samples) or winsorised residuals the
    statistic takes only a handful of distinct values; the empirical-quantile threshold then sits ON a tie, the
    strict rule `stat > h` never fires and the false-alarm rate is exactly 0 (the detector looks 'robust' only
    because it is dead). `jitter` adds U(0, jitter) noise to the statistic, i.e. a randomised test that breaks
    ties uniformly: the achieved false-alarm probability equals alpha and ordering of untied values is unchanged.
    `smooth=True` instead uses randomised (smoothed) p-values, exactly uniform under exchangeability but
    noisier; `jitter=0, smooth=False` reproduces the original tie-prone detector."""
    name = "D8_conformal"

    def __init__(self, w=5, smooth=False, jitter=1e-6, seed=12345):
        self.w, self.smooth, self.jitter, self.seed = w, smooth, jitter, seed

    def pvalues(self, R):
        P, M = R.Z.shape
        ncal = np.sum(np.isfinite(R.cal), axis=1)
        a = np.abs(R.Z)
        gt = np.zeros((P, M))
        eq = np.zeros((P, M))
        fin = np.isfinite(R.cal)
        for m in range(M):
            am = a[:, m][:, None]
            gt[:, m] = ((R.cal > am) & fin).sum(1)
            eq[:, m] = ((R.cal == am) & fin).sum(1)
        if self.smooth:
            U = np.random.default_rng(self.seed + 1).random((P, M))
            p = (gt + U * (eq + 1.0)) / (ncal[:, None] + 1.0)
        else:
            p = (1.0 + gt + eq) / (ncal[:, None] + 1.0)
        return np.clip(p, 1e-12, 1.0)

    def stat(self, R):
        lp = -np.log(self.pvalues(R))
        if self.jitter > 0:
            lp = lp + self.jitter * np.random.default_rng(self.seed).random(lp.shape)
        C = _csum(lp, R.active)
        return self._finish(_window_sum(C, self.w), R)


class MultiScale(Detector):
    """D9: multi-scale residual detector. Components (each normalised by its null quantile learned in
    fit(), then maximised): mean shift at scales w in (1, 4, 16); variance inflation at w in (4, 16);
    periodic power (least-squares sinusoid fit in a sliding window) at a bank of candidate periods.
    `use` selects component groups: 'mean', 'var', 'per' (used by the ablation study)."""
    name = "D9_multiscale"

    def __init__(self, mean_scales=(1, 4, 16), var_scales=(4, 16), per_window=24,
                 per_factors=(0.7, 0.85, 1.0, 1.15, 1.3), use=("mean", "var", "per"), q=0.95, label=None):
        self.mean_scales, self.var_scales = mean_scales, var_scales
        self.per_window, self.per_factors = per_window, per_factors
        self.use, self.q = tuple(use), q
        self.norm = None
        if label:
            self.name = label
        elif tuple(use) != ("mean", "var", "per"):
            self.name = "D9_multiscale[" + "+".join(use) + "]"

    def components(self, R):
        P, M = R.Z.shape
        comps = {}
        Z, A = R.Z, R.active
        C1 = _csum(Z, A)
        C2 = _csum(Z ** 2, A)
        if "mean" in self.use:
            for w in self.mean_scales:
                n = _count_window(A, w)
                comps[f"mean{w}"] = np.abs(_window_sum(C1, w)) / np.sqrt(np.maximum(n, 1))
        if "var" in self.use:
            for w in self.var_scales:
                n = _count_window(A, w)
                comps[f"var{w}"] = (_window_sum(C2, w) - n) / np.sqrt(2.0 * np.maximum(n, 1))
        if "per" in self.use:
            comps["per"] = self.periodic_power(R)
        return comps

    def periodic_power(self, R):
        """Max over candidate periods of the LS sinusoid-fit power Q = z'X(X'X)^-1 X'z in a sliding
        window of `per_window` observations (Q ~ chi2_2 under iid N(0,1))."""
        P, M = R.Z.shape
        Z, A, W = R.Z, R.active, self.per_window
        t = np.where(np.isfinite(R.Tm), R.Tm, 0.0)
        Zc = np.where(A, Z, 0.0)
        best = np.zeros((P, M))
        for f in self.per_factors:
            per = (R.period * f)[:, None]
            ph = 2 * np.pi * t / per
            c, s = np.cos(ph), np.sin(ph)
            ccs = {"cc": c * c, "ss": s * s, "cs": c * s, "zc": Zc * c, "zs": Zc * s}
            Cs = {k: _csum(v, A) for k, v in ccs.items()}
            Ws = {k: _window_sum(v, W) for k, v in Cs.items()}
            n = _count_window(A, W)
            det = Ws["cc"] * Ws["ss"] - Ws["cs"] ** 2
            ok = (n >= 8) & (det > 1e-3 * np.maximum(n, 1) ** 2)
            safe = np.where(ok, det, 1.0)
            Q = (Ws["ss"] * Ws["zc"] ** 2 - 2 * Ws["cs"] * Ws["zc"] * Ws["zs"] + Ws["cc"] * Ws["zs"] ** 2) / safe
            Q = np.where(ok, Q, 0.0)
            best = np.maximum(best, Q)
        return best

    def fit(self, R):
        comps = self.components(R)
        self.norm = {}
        for k, v in comps.items():
            v = np.where(R.active, v, -np.inf)
            mx = v.max(axis=1)
            mx = mx[np.isfinite(mx)]
            self.norm[k] = max(float(np.quantile(mx, self.q)), 1e-6) if mx.size else 1.0
        return self

    def stat(self, R):
        if self.norm is None:
            raise RuntimeError("MultiScale.fit() must be called on a no-failure calibration cohort first")
        comps = self.components(R)
        score = np.full(R.Z.shape, -np.inf)
        for k, v in comps.items():
            score = np.maximum(score, v / self.norm[k])
        return self._finish(np.where(np.isinf(score), 0.0, score), R)


class Hybrid(Detector):
    """D10: ensemble = max over member detectors of (statistic / null quantile)."""
    name = "D10_hybrid"

    def __init__(self, members=None, q=0.95, label=None):
        self.members = members if members is not None else [
            CUSUM(), GLR(), ShiryaevRoberts(), BOCPD(), ConformalAlarm(), MultiScale()]
        self.q = q
        if label:
            self.name = label
        self.norm = None

    def fit(self, R):
        self.norm = []
        for d in self.members:
            d.fit(R)
            s = d.stat_cached(R)
            mx = np.nanmax(np.where(np.isnan(s), -np.inf, s), axis=1)
            mx = mx[np.isfinite(mx)]
            self.norm.append(max(float(np.quantile(mx, self.q)), 1e-6) if mx.size else 1.0)
        return self

    def stat(self, R):
        out = np.full(R.Z.shape, -np.inf)
        for d, nrm in zip(self.members, self.norm):
            s = d.stat_cached(R) / nrm
            out = np.maximum(out, np.where(np.isnan(s), -np.inf, s))
        return self._finish(np.where(np.isinf(out), 0.0, out), R)


def all_detectors(include=None):
    ds = [Threshold(), MovingAverage(), EWMA(), CUSUM(), GLR(), ShiryaevRoberts(), BOCPD(),
          ConformalAlarm(), MultiScale(), Hybrid()]
    if include:
        ds = [d for d in ds if d.name.split("_")[0] in include or d.name in include]
    return ds


def ablation_detectors():
    """Detector variants for the ablation study (frequency features, multi-scale, conformal)."""
    return [
        MultiScale(use=("mean", "var"), label="D9_multiscale[-frequency]"),
        MultiScale(mean_scales=(4,), var_scales=(4,), use=("mean", "var"), label="D9_multiscale[single-scale]"),
        Hybrid(members=[CUSUM(), GLR(), ShiryaevRoberts(), BOCPD(), MultiScale()], label="D10_hybrid[-conformal]"),
        Hybrid(members=[CUSUM(), GLR(), ShiryaevRoberts(), BOCPD(), ConformalAlarm()], label="D10_hybrid[-multiscale]"),
    ]
