#!/usr/bin/env bash
# Standalone detached runner for the sigma0 bitrate sweep (B-lo 0.04 -> B-hi 0.16).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [ -z "${PROJECT_DIR:-}" ]; then
  if [ -f "${REPO_DIR}/../env.sh" ] || [ -d "${REPO_DIR}/../data" ]; then
    PROJECT_DIR="$(cd "${REPO_DIR}/.." && pwd)"
  else
    PROJECT_DIR="${REPO_DIR}"
  fi
fi
RUNS_DIR="${RUNS_DIR:-${PROJECT_DIR}/runs}"
mkdir -p "${RUNS_DIR}/B_lo_ul_s004" "${RUNS_DIR}/B_hi_ul_s016"
export PYTHONUNBUFFERED=1
cd "${REPO_DIR}"

echo "[$(date)] === [1/4] Training B-lo (sigma0=0.04, high bitrate) ==="
bash scripts/run_phase2.sh train_b_lo 2>&1 | tee -a "${RUNS_DIR}/B_lo_ul_s004/train.log"

echo "[$(date)] === [2/4] Evaluating B-lo (sigma0=0.04) ==="
bash scripts/run_phase2.sh eval_b_lo 2>&1 | tee -a "${RUNS_DIR}/B_lo_ul_s004/eval.log"

echo "[$(date)] === [3/4] Training B-hi (sigma0=0.16, low bitrate) ==="
bash scripts/run_phase2.sh train_b_hi 2>&1 | tee -a "${RUNS_DIR}/B_hi_ul_s016/train.log"

echo "[$(date)] === [4/4] Evaluating B-hi (sigma0=0.16) ==="
bash scripts/run_phase2.sh eval_b_hi 2>&1 | tee -a "${RUNS_DIR}/B_hi_ul_s016/eval.log"

echo "[$(date)] === BITRATE SWEEP COMPLETE ==="
