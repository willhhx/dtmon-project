import numpy as np
from dtmon.detectors import Residuals
from dtmon.newdet import CumBOCPD
from dtmon.calibrate import calibrate_threshold, per_patient_max


def _res(Z, burn=10):
    P, M = Z.shape
    Tm = np.tile(np.arange(M, dtype=float), (P, 1))
    return Residuals(Z, Tm, Tm >= burn, np.abs(Z[:, :burn]), float(burn), np.full(P, 6.0))


def test_cumbocpd_calibrates_and_detects():
    rng = np.random.default_rng(0)
    d = CumBOCPD()
    cal = _res(rng.normal(size=(300, 40)))
    h = calibrate_threshold(d.stat(cal), cal.Tm, 0.05)
    null = _res(rng.normal(size=(300, 40)))
    far = np.mean(per_patient_max(d.stat(null), null.Tm) > h)
    Zf = rng.normal(size=(300, 40))
    Zf[:, 25:] += 2.0
    flt = _res(Zf)
    s = d.stat(flt)
    power = np.mean(per_patient_max(s, flt.Tm) > h)
    assert np.isnan(s[:, :10]).all() and np.isfinite(s[:, 10:]).all()
    assert far < 0.12
    assert power > 0.8
