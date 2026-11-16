#!/usr/bin/env bash
# Confirmatory evaluation of D11 (cumulative Bayesian change-point detector). Run from the project root.
# Usage: bash scripts/run_newdet.sh [n_jobs]
set -e
J=${1:-4}
python3 -m dtmon.newdet bench    --n-patients 500 --seed 2026 --n-jobs "$J"
python3 -m dtmon.newdet transfer --n-patients 300 --seed 2026 --n-jobs "$J"
python3 -m dtmon.newdet real     --csv data/tumor.csv --k-burn 5 --n-splits 100
python3 -m dtmon.newdet decide
