"""Fault isolation: residual features at the alarm and classifiers F1-F4.

Fault classes: 1 parameter drift | 2 structural mismatch | 3 feedback-loop loss.
Feature groups: moments, acf (autocorrelation), spectrum (periodic power), state (EKF state-mismatch),
uncert (uncertainty inflation). Diagnosis uses K_PRE observations up to the alarm and K_POST after it.
"""
from __future__ import annotations
import numpy as np
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.covariance import LedoitWolf
from scipy.special import logsumexp
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from .calibrate import expected_calibration_error

try:                                   # optional dependency
    from xgboost import XGBClassifier
    HAVE_XGB = True
except Exception:                      # pragma: no cover
    HAVE_XGB = False

GROUPS = ("moments", "acf", "spectrum", "state", "uncert")
PER_FACTORS = (0.7, 0.85, 1.0, 1.15, 1.3)
SEQ_LEN = 24


def _window_idx(R, i, anchor, k_pre, k_post):
    valid = np.nonzero(np.isfinite(R.Z[i]) & (R.Tm[i] < np.inf))[0]
    if valid.size == 0 or anchor < 0:
        return np.array([], int)
    lo = max(anchor - k_pre + 1, 0)
    hi = min(anchor + k_post, valid.max())
    ii = np.arange(lo, hi + 1)
    ii = ii[np.isfinite(R.Z[i, ii]) & R.active[i, ii]]
    return ii


def _ls_power(z, t, period):
    ph = 2 * np.pi * t / period
    X = np.stack([np.cos(ph), np.sin(ph)], axis=1)
    try:
        G = X.T @ X + 1e-6 * np.eye(2)
        b = np.linalg.solve(G, X.T @ z)
        return float(b @ G @ b)
    except np.linalg.LinAlgError:
        return 0.0


def feature_names(groups=GROUPS):
    names = []
    if "moments" in groups:
        names += ["mean", "std", "skew", "kurt", "mean_abs", "max_abs", "slope", "mean_nu_rel", "std_nu_rel"]
    if "acf" in groups:
        names += ["acf1", "acf2"]
    if "spectrum" in groups:
        names += ["per_q1", "per_qmax", "per_argmax"]
    if "state" in groups:
        names += ["corr_mean", "corr_max", "corr_rel"]
    if "uncert" in groups:
        names += ["infl_mean", "infl_rel"]
    return names


def extract_features(R, anchors, k_pre=20, k_post=10, groups=GROUPS, min_obs=6):
    """Feature matrix (P, F) for the window around each patient's anchor observation (idx, -1 = none).
    Returns (X, ok) where ok marks patients with enough observations."""
    P = R.P
    names = feature_names(groups)
    X = np.zeros((P, len(names)))
    ok = np.zeros(P, bool)
    corr_a, infl_a, nu_a = R.aux.get("corr"), R.aux.get("infl"), R.aux.get("nu_rel")
    for i in range(P):
        ii = _window_idx(R, i, int(anchors[i]), k_pre, k_post)
        if ii.size < min_obs:
            continue
        ok[i] = True
        z, t = R.Z[i, ii], R.Tm[i, ii]
        f = []
        if "moments" in groups:
            m, s = z.mean(), z.std() + 1e-9
            sk = np.mean(((z - m) / s) ** 3)
            ku = np.mean(((z - m) / s) ** 4) - 3
            slope = np.polyfit(np.arange(z.size), z, 1)[0]
            nu = nu_a[i, ii] if nu_a is not None else np.zeros(z.size)
            f += [m, s, sk, ku, np.abs(z).mean(), np.abs(z).max(), slope, np.nanmean(nu), np.nanstd(nu)]
        if "acf" in groups:
            zc = z - z.mean()
            den = np.sum(zc ** 2) + 1e-9
            f += [np.sum(zc[1:] * zc[:-1]) / den, np.sum(zc[2:] * zc[:-2]) / den if z.size > 3 else 0.0]
        if "spectrum" in groups:
            qs = [_ls_power(z, t, R.period[i] * pf) / z.size for pf in PER_FACTORS]
            f += [qs[2], max(qs), float(np.argmax(qs))]
        if "state" in groups:
            if corr_a is not None:
                c = corr_a[i, ii]
                burn = corr_a[i][np.isfinite(corr_a[i]) & (R.Tm[i] < R.burn_in)]
                cb = np.nanmean(burn) if burn.size else np.nan
                f += [np.nanmean(c), np.nanmax(c), np.nanmean(c) / (cb + 1e-6) if np.isfinite(cb) else 1.0]
            else:
                f += [0.0, 0.0, 1.0]
        if "uncert" in groups:
            if infl_a is not None:
                lg = np.log(np.maximum(infl_a[i, ii], 1e-6))
                burn = infl_a[i][np.isfinite(infl_a[i]) & (R.Tm[i] < R.burn_in)]
                lb = np.mean(np.log(np.maximum(burn, 1e-6))) if burn.size else np.nan
                f += [np.nanmean(lg), np.nanmean(lg) - (lb if np.isfinite(lb) else np.nanmean(lg))]
            else:
                f += [0.0, 0.0]
        X[i] = np.nan_to_num(np.array(f, float), nan=0.0, posinf=0.0, neginf=0.0)
    return X, ok


def extract_sequences(R, anchors, k_pre=SEQ_LEN, k_post=0):
    """Fixed-length residual sequences (last SEQ_LEN residuals up to the anchor + log gaps) for F4."""
    P = R.P
    S = np.zeros((P, 2 * SEQ_LEN))
    ok = np.zeros(P, bool)
    for i in range(P):
        ii = _window_idx(R, i, int(anchors[i]), SEQ_LEN, k_post)
        if ii.size < 6:
            continue
        ii = ii[-SEQ_LEN:]
        ok[i] = True
        z = R.Z[i, ii]
        gaps = np.log1p(np.diff(R.Tm[i, ii], prepend=R.Tm[i, ii[0]]))
        S[i, SEQ_LEN - z.size:SEQ_LEN] = z
        S[i, 2 * SEQ_LEN - z.size:] = gaps
    return S, ok


class GaussianLikelihoodClassifier(BaseEstimator, ClassifierMixin):
    """F3: class-conditional multivariate-Gaussian likelihood classifier with Ledoit-Wolf shrunk covariances
    (works with few samples per class) and empirical priors. predict_proba = posterior from the likelihoods."""

    def __init__(self, ridge=1e-3):
        self.ridge = ridge

    def fit(self, X, y):
        X = np.asarray(X, float)
        self.classes_ = np.unique(y)
        self.mu_, self.prec_, self.logdet_, self.logprior_ = [], [], [], []
        for c in self.classes_:
            Xc = X[np.asarray(y) == c]
            mu = Xc.mean(0)
            if len(Xc) >= 3:
                cov = LedoitWolf().fit(Xc).covariance_
            else:
                cov = np.cov(X.T) if len(X) > 2 else np.eye(X.shape[1])
            cov = np.atleast_2d(cov) + self.ridge * np.eye(X.shape[1])
            sign, ld = np.linalg.slogdet(cov)
            self.mu_.append(mu)
            self.prec_.append(np.linalg.inv(cov))
            self.logdet_.append(ld)
            self.logprior_.append(np.log(len(Xc) / len(X)))
        return self

    def _joint(self, X):
        X = np.asarray(X, float)
        L = []
        for mu, P, ld, lp in zip(self.mu_, self.prec_, self.logdet_, self.logprior_):
            d = X - mu
            L.append(-0.5 * np.einsum("ij,jk,ik->i", d, P, d) - 0.5 * ld + lp)
        return np.stack(L, axis=1)

    def predict_proba(self, X):
        J = self._joint(X)
        return np.exp(J - logsumexp(J, axis=1, keepdims=True))

    def predict(self, X):
        return self.classes_[np.argmax(self._joint(X), axis=1)]


# ----------------------------------------------------------------------------------------------
def make_classifier(name: str, seed: int = 0):
    if name == "F1_rf":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=seed, n_jobs=1)
    if name == "F2_xgb":
        if HAVE_XGB:
            return XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.08, subsample=0.9,
                                 colsample_bytree=0.9, random_state=seed, n_jobs=1, eval_metric="mlogloss")
        return HistGradientBoostingClassifier(max_depth=4, learning_rate=0.08, max_iter=250, random_state=seed)
    if name == "F3_likelihood":
        return make_pipeline(StandardScaler(), GaussianLikelihoodClassifier())
    if name == "F4_neural":
        return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-2,
                             early_stopping=True, max_iter=400, random_state=seed))
    raise ValueError(name)


CLASSIFIERS = ("F1_rf", "F2_xgb", "F3_likelihood", "F4_neural")


def cross_validate_diagnosis(X, y, groups, name, n_splits=5, seed=0):
    """Patient-grouped stratified CV. Returns dict with accuracy, macro-F1, confusion matrix, ECE."""
    y = np.asarray(y)
    classes = np.unique(y)
    if classes.size < 2 or len(y) < 4 * classes.size:
        return None
    n_splits = int(max(2, min(n_splits, np.min([np.sum(y == c) for c in classes]))))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pred = np.empty(len(y), dtype=y.dtype)
    proba = np.zeros((len(y), classes.size))
    for tr, te in cv.split(X, y, groups):
        clf = make_classifier(name, seed)
        # xgboost needs 0..K-1 labels
        ymap = {c: i for i, c in enumerate(classes)}
        clf.fit(X[tr], np.array([ymap[c] for c in y[tr]]))
        p = clf.predict_proba(X[te])
        full = np.zeros((len(te), classes.size))
        full[:, clf.classes_ if hasattr(clf, "classes_") else clf[-1].classes_] = p
        proba[te] = full
        pred[te] = classes[full.argmax(1)]
    return dict(accuracy=float(accuracy_score(y, pred)), macro_f1=float(f1_score(y, pred, average="macro")),
                confusion=confusion_matrix(y, pred, labels=classes), classes=classes,
                ece=expected_calibration_error(proba, y, classes), backend=("xgboost" if (name == "F2_xgb" and HAVE_XGB) else "sklearn"))
