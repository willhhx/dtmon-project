"""Unit tests (run:  python -m unittest discover -s tests -v   or   pytest -q tests)."""
import os, sys, unittest
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dtmon.models import get_model, FaultSpec
from dtmon.simulate import make_population, make_fault, simulate_truth, observe
from dtmon.filter import run_ekf
from dtmon.sampling import make_mask
from dtmon.detectors import (Residuals, prepare_residuals, all_detectors, ShiryaevRoberts, CUSUM, BOCPD,
                             ConformalAlarm, MultiScale, Threshold)
from dtmon.calibrate import calibrate_threshold, evaluate, summarize, first_alarm, auroc, OUT_DET, OUT_FA
from dtmon.diagnose import extract_features, feature_names, make_classifier
from dtmon import theory


def synth(P, M, kind="none", seed=0, burn=20, chg=140, mag=1.0):
    r = np.random.default_rng(seed)
    Z = r.normal(size=(P, M))
    Tm = np.tile(np.arange(M, dtype=float), (P, 1))
    post = Tm >= chg
    if kind == "mean":
        Z = Z + mag * post
    elif kind == "var":
        Z = np.where(post, Z * (1 + mag), Z)
    elif kind == "per":
        Z = Z + mag * np.sin(2 * np.pi * Tm / 12.0) * post
    active = Tm >= burn
    return Residuals(Z, Tm, active, np.abs(Z[:, :burn]), float(burn), np.full(P, 12.0))


class TestModels(unittest.TestCase):
    def test_jacobian_matches_finite_difference(self):
        rng = np.random.default_rng(0)
        for name in ("glucose", "tumor"):
            m = get_model(name)
            pop = make_population(m, 5, 1)
            x = m.initial_state(pop.params) * (1 + 0.1 * rng.random((5, m.n)))
            u = np.stack([np.full(5, 0.3)] * m.n_inputs, axis=1)
            J = m.jac_twin(x, u, pop.params)
            for j in range(m.n):
                e = np.zeros(m.n); e[j] = 1e-6
                fd = (m.f_twin(x + e, u, pop.params) - m.f_twin(x - e, u, pop.params)) / 2e-6
                np.testing.assert_allclose(J[:, :, j], fd, rtol=1e-3, atol=1e-5)

    def test_no_fault_is_identity_and_fault_changes_trajectory(self):
        m = get_model("glucose")
        pop = make_population(m, 4, 2)
        y0 = simulate_truth(pop, make_fault(m, pop, 0))
        y0b = simulate_truth(pop, FaultSpec.none(4))
        np.testing.assert_allclose(y0, y0b)
        for kind in (1, 2, 3):
            y = simulate_truth(pop, make_fault(m, pop, kind, "large", sub=0))
            i0 = int((m.t_change - 5) / m.dt)
            np.testing.assert_allclose(y[:, :i0], y0[:, :i0], atol=1e-4)        # unchanged before onset
            self.assertGreater(np.abs(y[:, -400:] - y0[:, -400:]).max(), 0.5)   # changed after

    def test_all_structure_subtypes_run_and_are_finite(self):
        for name in ("glucose", "tumor"):
            m = get_model(name)
            pop = make_population(m, 6, 3)
            for sub in range(4):
                y = simulate_truth(pop, make_fault(m, pop, 2, "large", sub=sub))
                self.assertTrue(np.isfinite(y).all(), (name, sub))

    def test_common_random_numbers(self):
        m = get_model("tumor")
        a = make_population(m, 5, 9); b = make_population(m, 5, 9)
        for k in a.params:
            np.testing.assert_array_equal(a.params[k], b.params[k])


class TestFilterAndSampling(unittest.TestCase):
    def test_null_innovations_are_roughly_standard_normal(self):
        for name, nz in (("glucose", 0.2), ("tumor", 0.2)):
            m = get_model(name)
            pop = make_population(m, 40, 4, cal_sigma=0.0)           # perfectly calibrated twin
            f = make_fault(m, pop, 0)
            co = observe(m, pop, simulate_truth(pop, f), f, "moderate", nz, 5)
            z = run_ekf(m, pop, co).z
            zz = z[:, int(m.burn_in / m.dt):]
            zz = zz[np.isfinite(zz)]
            self.assertLess(abs(np.mean(zz)), 0.35, name)
            self.assertTrue(0.6 < np.std(zz) < 1.5, (name, np.std(zz)))

    def test_masks(self):
        m = get_model("glucose")
        rng = np.random.default_rng(0)
        for sc in ("dense", "moderate", "sparse", "realistic"):
            mk = make_mask(m, sc, 10, rng)
            self.assertEqual(mk.shape, (10, m.n_steps + 1))
            self.assertTrue(mk.sum(1).min() >= 1)
        d = make_mask(m, "dense", 3, rng).sum(1).mean()
        s = make_mask(m, "sparse", 3, rng).sum(1).mean()
        self.assertGreater(d, 20 * s)
        miss = make_mask(m, "dense", 20, rng, miss_prob=0.5).sum(1).mean()
        self.assertTrue(0.35 * d < miss < 0.65 * d)


class TestDetectors(unittest.TestCase):
    def test_false_alarm_calibration_all_detectors(self):
        cal, ev = synth(600, 200, seed=1), synth(600, 200, seed=2)
        for D in all_detectors():
            D.fit(cal)
            h = calibrate_threshold(D.stat(cal), cal.Tm, 0.05)
            s = D.stat(ev)
            far = np.mean(np.nanmax(np.where(np.isnan(s), -np.inf, s), axis=1) > h)
            self.assertLess(abs(far - 0.05), 0.045, (D.name, far))

    def test_power_ordering_on_known_shifts(self):
        cal = synth(500, 260, seed=1)
        D = {d.name: d for d in all_detectors(include=("D1", "D4", "D9"))}
        for d in D.values():
            d.fit(cal)
        h = {n: calibrate_threshold(d.stat(cal), cal.Tm, 0.05) for n, d in D.items()}
        def det_rate(kind, name):
            R = synth(500, 260, kind, seed=3)
            s = D[name].stat(R)
            ev = evaluate(s, R.Tm, h[name], np.full(500, 140.0), 260.0)
            return summarize(ev)["detect_rate"]
        self.assertGreater(det_rate("mean", "D4_cusum"), det_rate("mean", "D1_threshold") + 0.4)
        self.assertGreater(det_rate("per", "D9_multiscale"), det_rate("per", "D4_cusum") + 0.3)
        self.assertGreater(det_rate("var", "D1_threshold"), 0.8)

    def test_sr_threshold_theorem(self):
        for r in theory.check_sr_false_alarm(N=150, A_list=(200, 1000), n_trials=2000):
            self.assertTrue(r["holds"], r)

    def test_bocpd_detects_big_shift(self):
        R = synth(100, 200, "mean", seed=4, mag=3.0)
        s = BOCPD().stat(R)
        self.assertGreater(np.nanmax(s[:, 140:150]), np.nanmax(s[:, 40:130]) + 1.0)

    def test_conformal_pvalue_validity(self):
        for r in theory.check_conformal_validity(n_rep=100):
            self.assertTrue(r["valid"], r)

    def test_multiscale_requires_fit(self):
        with self.assertRaises(RuntimeError):
            MultiScale().stat(synth(10, 60))

    def test_padding_is_ignored(self):
        R = synth(20, 100, "mean", seed=5)
        R.Z[:, 80:] = np.nan; R.Tm[:, 80:] = np.inf; R.active[:, 80:] = False
        for D in all_detectors(include=("D2", "D3", "D4", "D5", "D6", "D8")):
            D.fit(R)
            s = D.stat(R)
            self.assertTrue(np.isnan(s[:, 80:]).all(), D.name)


class TestCalibrationAndDiagnosis(unittest.TestCase):
    def test_first_alarm_and_outcomes(self):
        stat = np.array([[0, 1, 5, 2], [0, 0, 0, 0.0], [9, 0, 0, 0]])
        Tm = np.tile(np.array([1.0, 2, 3, 4]), (3, 1))
        t, idx = first_alarm(stat, Tm, 3.0)
        self.assertEqual(list(t), [3.0, np.inf, 1.0])
        ev = evaluate(stat, Tm, 3.0, np.array([2.5, 2.5, 2.5]), 4.0)
        self.assertEqual(list(ev["outcome"]), [OUT_DET, 0, OUT_FA])

    def test_auroc_perfect_and_chance(self):
        self.assertEqual(auroc(np.array([5, 6, 7.0]), np.array([1, 2, 3.0])), 1.0)
        self.assertAlmostEqual(auroc(np.arange(100.0), np.arange(100.0)), 0.5, places=1)

    def test_feature_extraction_shapes_and_classifiers(self):
        R = synth(30, 100, "mean", seed=6)
        R.aux = {"corr": np.abs(R.Z), "infl": np.ones_like(R.Z) * 2, "nu_rel": R.Z * 0.1}
        anchors = np.full(30, 60)
        X, ok = extract_features(R, anchors)
        self.assertEqual(X.shape, (30, len(feature_names())))
        self.assertTrue(ok.all() and np.isfinite(X).all())
        y = np.arange(30) % 3
        for name in ("F1_rf", "F2_xgb", "F3_likelihood", "F4_neural"):
            clf = make_classifier(name)
            clf.fit(X, y)
            self.assertEqual(clf.predict_proba(X).shape, (30, 3))


class TestEndToEnd(unittest.TestCase):
    def test_tiny_condition_runs(self):
        from dtmon.experiments import run_condition
        res = run_condition(dict(model="tumor", noise=0.1, scheme="realistic"), n_patients=24, seed=0,
                            levels=("large",), verbose=False)
        S = res["summary"]
        self.assertEqual(set(S.detector), {d.name for d in all_detectors()})
        self.assertTrue((S.detect_rate >= 0).all() and (S.auroc.dropna() >= 0).all())
        self.assertTrue(np.isfinite(list(res["thresholds"].values())).all())


if __name__ == "__main__":
    unittest.main()
