"""Tests for the paper-strengthening code (python -m unittest discover -s tests -v)."""
import os, sys, tempfile, unittest
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dtmon.detectors import Residuals, BOCPD, ConformalAlarm, CUSUM, ShiryaevRoberts
from dtmon.calibrate import calibrate_threshold, far_full, per_patient_max
from dtmon.audit import audit_results, tie_fraction
from dtmon.extra import wilson, _acc_ci, default_shifts, power_transfer
from dtmon import realdata


def synth(P, M, burn, chg, mag=0.0, seed=0):
    r = np.random.default_rng(seed)
    Z = r.normal(size=(P, M))
    Tm = np.tile(np.arange(M, dtype=float), (P, 1))
    Z = Z + mag * (Tm >= chg)
    return Residuals(Z, Tm, Tm >= burn, np.abs(Z[:, :burn]).copy(), float(burn), np.full(P, 12.0))


class TestDetectorFixes(unittest.TestCase):
    def test_bocpd_not_degenerate_with_few_observations(self):
        """Regression: with < 20 observations the BOCPD statistic used to sit at its ceiling for every patient."""
        cal, te = synth(1500, 25, 3, 12, 0, 1), synth(1500, 25, 3, 12, 0, 2)
        d = BOCPD()
        s = d.stat(cal)
        mx = per_patient_max(s, cal.Tm)
        self.assertGreater(len(np.unique(mx)), 1000)                      # not a ceiling
        h = calibrate_threshold(s, cal.Tm, 0.05)
        self.assertTrue(0.02 < far_full(d.stat(te), te.Tm, h) < 0.09)
        fa = synth(1500, 25, 3, 12, 1.5, 3)
        pw = np.mean(per_patient_max(d.stat(fa), fa.Tm, t_from=12) > h)
        self.assertGreater(pw, 0.12)                                       # has power

    def test_conformal_ties_fixed(self):
        """Regression: with 3 burn-in scores the old statistic was tied, FAR = 0 and power = 0."""
        cal, te = synth(1500, 25, 3, 12, 0, 1), synth(1500, 25, 3, 12, 0, 2)
        old = ConformalAlarm(smooth=False, jitter=0.0)
        h_old = calibrate_threshold(old.stat(cal), cal.Tm, 0.05)
        self.assertLess(far_full(old.stat(te), te.Tm, h_old), 0.01)        # dead
        new = ConformalAlarm()
        h = calibrate_threshold(new.stat(cal), cal.Tm, 0.05)
        far = far_full(new.stat(te), te.Tm, h)
        self.assertTrue(0.025 < far < 0.075, far)                          # calibrated

    def test_jitter_does_not_change_untied_order(self):
        R = synth(50, 40, 10, 20, 0, 5)
        a, b = ConformalAlarm(jitter=0.0).stat(R), ConformalAlarm(jitter=1e-6).stat(R)
        m = np.isfinite(a)
        self.assertLess(np.nanmax(np.abs(a - b)), 1e-4)


class TestAudit(unittest.TestCase):
    def test_flags_dead_detector(self):
        rows = []
        for det, far, pw in [("D4_cusum", 0.05, 0.6), ("D7_bocpd", 0.0, 0.0), ("D8_conformal", 0.05, 0.01)]:
            for cid in ("a", "b"):
                for lv in ("medium", "large"):
                    rows.append(dict(cond_id=cid, model="glucose", scheme="moderate", noise=0.1, detector=det,
                                     far_null_full=far, n_patients=500, variant="full", level=lv,
                                     detect_rate_nofa=pw))
        A = audit_results(pd.DataFrame(rows), alpha=0.05)
        f = dict(zip(A.detector, A.flag))
        self.assertEqual(f["D4_cusum"], "")
        self.assertIn("DEAD_FAR", f["D7_bocpd"])
        self.assertIn("NO_POWER", f["D8_conformal"])

    def test_tie_fraction(self):
        s = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 0.0]])
        self.assertAlmostEqual(tie_fraction(s, np.zeros_like(s), 2.0), 2 / 3)


class TestStats(unittest.TestCase):
    def test_wilson(self):
        lo, hi = wilson(0, 100)
        self.assertEqual(lo, 0.0)
        self.assertTrue(0.03 < hi < 0.05)
        lo, hi = wilson(50, 100)
        self.assertTrue(0.40 < lo < 0.41 and 0.59 < hi < 0.60)

    def test_cluster_bootstrap_ci_contains_estimate(self):
        rng = np.random.default_rng(0)
        gid = np.repeat(np.arange(100), 3)
        ok = rng.random(300) < 0.8
        a, lo, hi = _acc_ci(ok, gid, 300)
        self.assertTrue(lo <= a <= hi)


class TestPowerTransfer(unittest.TestCase):
    def test_runs_and_reports_power(self):
        sh = [s for s in default_shifts() if s[0] in ("nominal", "param error 15%")]
        f, p, s = power_transfer("tumor", shifts=sh, n_patients=12, seed=0, detectors=[CUSUM(), ConformalAlarm()],
                                 kinds=(1,), levels=("large",))
        self.assertEqual(set(s.detector), {"D4_cusum", "D8_conformal"})
        for c in ("far", "far_lo", "far_hi", "power_nominal_thr", "power_oracle_thr"):
            self.assertIn(c, s.columns)
        self.assertTrue(((s.far >= 0) & (s.far <= 1)).all())


class TestRealData(unittest.TestCase):
    def test_demo_pipeline(self):
        with tempfile.TemporaryDirectory() as d:
            path = realdata.make_demo_csv(os.path.join(d, "x.csv"), n_patients=60, seed=1)
            series = realdata.load_series(path, min_obs=9)
            far, pw, info = realdata.run_realdata(series, k_burn=5, n_splits=3, mags=(6.0,), kinds=("variance",),
                                                  detectors=[CUSUM(), ShiryaevRoberts()])
        self.assertEqual(info["n_patients"], 60)
        self.assertTrue((far.far <= 1).all())
        self.assertEqual(len(pw), 2)

    def test_gompertz_fit_recovers_curve(self):
        t = np.linspace(0, 30, 12)
        p = np.array([np.log(30.0), 0.25, 0.04])
        y = realdata.gompertz_log(t, p)
        q, s, ok = realdata.fit_twin(t, y)
        self.assertTrue(ok)
        self.assertLess(np.max(np.abs(realdata.gompertz_log(t, q) - y)), 1e-3)


if __name__ == "__main__":
    unittest.main()
