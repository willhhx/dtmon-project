#!/usr/bin/env bash
# Full reproduction for the paper. Run from the project root (the folder that contains dtmon/).
#   bash scripts/reproduce_paper.sh            # everything (roughly 30-60 min on 4 cores)
#   bash scripts/reproduce_paper.sh quick      # smoke test with tiny cohorts (about 5 min)
# Every step is resumable (finished conditions are skipped), so it is safe to re-run after an interruption.
set -euo pipefail
cd "$(dirname "$0")/.."
JOBS="${JOBS:-4}"
if [[ "${1:-}" == "quick" ]]; then PM=40; PA=30; PT=40; OUT=results_quick; QUICK=1; MAIN_ARGS="--noises 0.1 --schemes moderate realistic"
else PM=500; PA=200; PT=300; OUT=results; QUICK=0; MAIN_ARGS=""; fi
PY="${PYTHON:-python}"
step() { echo; echo "=== $* ==="; }

step "0. unit tests (must pass)"
$PY -m unittest discover -s tests

step "0b. refuse to reuse results produced by the buggy detectors (runs are resumable, so stale files would be reused!)"
for d in "$OUT/main_P$PM" "$OUT/ablation_P$PA" "$OUT/stress_P$PA"; do
  if [[ -d "$d" ]] && [[ -n "$(find "$d" -name 'summary_*.csv.gz' ! -newer dtmon/detectors.py -print -quit)" ]]; then
    echo "STALE results in $d are older than the detector fixes. Delete them and re-run:  rm -rf \"$d\""; exit 1
  fi
done

step "1. main grid (2 models x 4 noise x 4 sampling schemes)  [~25-40 min at 500 patients]"
$PY -m dtmon main --n-patients $PM --n-jobs $JOBS --out $OUT $MAIN_ARGS

step "2. SANITY GATE: flag degenerate detectors (dead / liberal / no power). Read this before anything else."
$PY -m dtmon audit --results $OUT/main_P$PM --out $OUT || echo "(audit flagged detectors: read the table above before interpreting results)"

step "3. headline analysis: verdict, scorecard, diagnosis tables, figures"
$PY -m dtmon decide    --results $OUT/main_P$PM --out $OUT/main_analysis
$PY -m dtmon diagnosis --results $OUT/main_P$PM --out $OUT/main_analysis
$PY -m dtmon figures   --results $OUT/main_P$PM --out $OUT/main_analysis

step "4. ablation and stress runs (skipped in quick mode)"
if [[ "$QUICK" == "0" ]]; then
  $PY -m dtmon ablation --n-patients $PA --n-jobs $JOBS --out $OUT
  $PY -m dtmon stress   --n-patients $PA --n-jobs $JOBS --out $OUT
  $PY -m dtmon decide   --results $OUT/ablation_P$PA --out $OUT/ablation_analysis
fi

step "5. H3 properly: false alarms AND power under misspecification (calibrate nominal, deploy shifted)"
$PY -m dtmon transfer2 --n-patients $PT --n-jobs $JOBS --out $OUT

step "6. price of conformal calibration: power vs burn-in length"
$PY -m dtmon burnin --n-patients $PA --out $OUT --models tumor glucose

step "7. fault isolation vs baselines, end-to-end accuracy, per-subtype recall"
$PY -m dtmon diag2 --results $OUT/main_P$PM --out $OUT

step "8. theory checks"
$PY -m dtmon theory --out $OUT

step "9. real-data validation (needs your csv with columns id,time,value; skipped if REAL_CSV is unset)"
if [[ -n "${REAL_CSV:-}" ]]; then
  $PY -m dtmon realdata --csv "$REAL_CSV" --n-splits 100 --out $OUT
else
  $PY -m dtmon realdata --demo --out $OUT      # synthetic series: pipeline test only, NOT evidence
  echo "Set REAL_CSV=path/to/tumour_series.csv to run on real data."
fi
echo; echo "Done. Key outputs: $OUT/audit  $OUT/main_analysis  $OUT/transfer2  $OUT/burnin  $OUT/diag_baselines  $OUT/realdata"
