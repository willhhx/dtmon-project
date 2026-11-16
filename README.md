# dtmon - detecting and diagnosing failure in patient-specific digital twins

Quickest change detection with fault isolation for sparsely sampled mechanistic models.
Two published ODE benchmarks (glucose-insulin: Bergman; tumour-immune: Kuznetsov), three injected failure types
(parameter drift, structural mismatch, feedback-loop loss), an observer-based (EKF) twin, ten sequential detectors,
fault-isolation classifiers, equal-false-alarm evaluation, ablations, stress tests and GREEN/YELLOW/RED decision logic.

> Parameter values are representative literature values jittered across virtual patients. They are NOT clinically
> validated. Do not make biological claims from this code without calibrating the models.

## Install (Python 3.10+)
```
pip install -r requirements.txt          # xgboost is optional (falls back to scikit-learn boosting)
python -m unittest discover -s tests -v  # 20 tests, ~1 min
```

## Pipeline in one paragraph
For every *condition* (model, noise CV, sampling scheme, missingness, calibration error, filter variant) the code
(1) simulates an independent no-failure calibration cohort, runs the twin filter, fits the detectors and calibrates every
detector's threshold to the same false-alarm probability alpha (default 0.05 over the whole monitored window);
(2) simulates an evaluation population and, on the SAME patients (common random numbers), a no-failure cohort plus failure
cohorts (3 types x small/medium/large); (3) records per-patient alarm times, delays and scores, AUROC against the paired
null, and diagnosis features at each detector's alarm. Everything is written to csv.gz / npz so analysis and figures can
be regenerated without re-simulating. Runs are resumable (finished conditions are skipped).

## Commands
All commands: `python -m dtmon <command> [options]`. Common options: `--out results --seed 0 --alpha 0.05
--n-patients 100 --n-jobs 1 --models glucose tumor`.

| Command | What it does | Status when written |
|---|---|---|
| `pilot --n-patients 100` | per model: 4 sampling schemes + 2 filter-variant conditions (12 conditions for 2 models), all detectors + ablation detectors | an earlier layout (dense/sparse/realistic + variants at dense, 60 patients) was run; the shipped `results/pilot` comes from it |
| `decide --results results/pilot` | paired comparison vs strongest baseline, GREEN/YELLOW/RED, hypothesis scorecard, ablation tables | run |
| `diagnosis --results results/pilot` | fault-isolation tables (F1-F4), feature-group ablation | run |
| `figures --results results/pilot` | Figures 1-8 (PNG + PDF) into `results/figures` | run |
| `theory` | numerical checks of T1-T5 (docs/theory.md) | run |
| `main --n-patients 500 --n-jobs 4` | full grid: 2 models x 4 noise (5/10/20/40%) x 4 sampling schemes | run only at tiny scale (30 patients, 4 conditions) |
| `ablation --n-patients 200 --n-jobs 4` | filter variants (no normalisation, open-loop twin) + detector ablations | run only at tiny scale (20 patients) |
| `stress --n-patients 200 --n-jobs 4` | noise 5-40%, missing data 20/40%, calibration error 2-20%, misspecified noise model x0.5/x2 | run only at tiny scale (20 patients, tumour) |
| `transfer --n-patients 200` | H3: calibrate at nominal conditions, measure false alarms under shifted conditions | run only at tiny scale (60 patients, tumour) |
| `all` | pilot -> theory -> decide -> diagnosis -> figures | not run as one command |
| `audit --results DIR` | sanity gate: flags detectors that never alarm (DEAD_FAR), are liberal, or have no power | new (tested at tiny scale) |
| `transfer2 --n-patients 300 --n-jobs 4` | H3 with power: false alarms AND power (nominal and re-calibrated threshold) under 7 shifts | new (tested at tiny scale) |
| `burnin --n-patients 200` | detection rate vs number of burn-in observations (price of conformal calibration) | new (tested at tiny scale) |
| `diag2 --results DIR` | fault isolation vs baselines, end-to-end accuracy, per-class/level/subtype recall with CIs | new (tested at tiny scale) |
| `realdata --csv FILE` / `--demo` | semi-synthetic validation on real longitudinal series (csv: id,time,value) | new (demo tested; run it on real data) |
| `bash scripts/reproduce_paper.sh [quick]` | the whole paper pipeline in order | new |

Use `--n-patients 100`, `500`, `1000` for the three cohort sizes. Single-core cost (rough, measured at 20-80 patients,
scales ~linearly): glucose dense ~0.5 s/patient/condition, tumour ~0.2 s/patient/condition; `main` at 500 patients is
roughly 1-2 h on one core, 1000 patients ~3 h; use `--n-jobs`.

Typical full workflow:
```
python -m dtmon main      --n-patients 500 --n-jobs 4 --out results
python -m dtmon ablation  --n-patients 200 --n-jobs 4 --out results
python -m dtmon stress    --n-patients 200 --n-jobs 4 --out results
python -m dtmon transfer  --n-patients 200 --out results
python -m dtmon decide    --results results/main_P500 --out results
python -m dtmon decide    --results results/ablation_P200 --out results/ablation_analysis
python -m dtmon diagnosis --results results/main_P500 --out results
python -m dtmon figures   --results results/main_P500 --out results
python -m dtmon theory    --out results
```
Note: `decide`/`diagnosis`/`figures` write into `<out>/analysis` and `<out>/figures`; use different `--out` values to keep
results of different runs apart.

## Layout
```
dtmon/models.py       ODE models, analytic Jacobians, failure injection (truth only; the twin never sees a failure)
dtmon/sampling.py     dense / moderate / sparse / realistic (irregular) schemes, random and block missingness
dtmon/simulate.py     virtual patients (common random numbers), failure specs, noisy sparse observations
dtmon/filter.py       batched EKF twin -> standardised innovations, state-mismatch and uncertainty-inflation signals
dtmon/detectors.py    D1 threshold, D2 moving avg, D3 EWMA, D4 CUSUM, D5 GLR, D6 mixture Shiryaev-Roberts,
                      D7 BOCPD, D8 conformal, D9 multi-scale, D10 hybrid (+ ablation variants)
dtmon/calibrate.py    equal-false-alarm thresholds, outcomes, AUROC, paired bootstrap, calibration error
dtmon/diagnose.py     features (moments, ACF, spectrum, state mismatch, uncertainty) and classifiers F1-F4
dtmon/experiments.py  condition runner, grids, resumable parallel runs, transfer (misspecification) test
dtmon/analysis.py     comparisons, diagnosis experiments, ablations, hypothesis scorecard, decision logic
dtmon/theory.py       formulas and numerical verification of T1-T5
dtmon/figures.py      Figures 1-8
dtmon/audit.py        sanity gate (dead / liberal / no-power detectors)
dtmon/extra.py        transfer2 (FAR + power), burn-in curve, diagnosis baselines
dtmon/realdata.py     semi-synthetic real-data validation (Gompertz twin + Kalman offset + injected failures)
dtmon/figures_extra.py  figures for the above
dtmon/cli_paper.py    CLI for the above
scripts/              reproduce_paper.sh
docs/                 theory.md (proofs), hypotheses.md, literature_review.md
literature/           lit_search.py, literature_matrix.csv (seeded, see caveats)
tests/                unit tests
results/              pilot outputs (60 patients) as a worked example
```
