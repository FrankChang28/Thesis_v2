#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${SUBJECT_NIRS_PYTHON:-python}"
export PYTHONPATH="./src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg
if [[ "${1:-}" == *.yaml ]]; then
    "$PYTHON" scripts/train_stage2.py --config "$1"
else
    "$PYTHON" scripts/run_subject_cohort.py --phase stage2 --execute "$@"
fi
