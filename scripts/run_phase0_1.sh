#!/usr/bin/env bash
# Reference commands for Phase 0 (pixel baseline A0) and Phase 1 (latent caching + frozen SD-VAE A1).
# Usage: bash scripts/run_phase0_1.sh <stage>
# Stages: cache_sdvae | cache_dinov2 | train_a0_uni | train_a0_cc | eval_a0_uni | train_a1_vae | eval_a1_vae
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
[ -f "${PROJECT_DIR}/env.sh" ] && source "${PROJECT_DIR}/env.sh"
cd "${REPO_DIR}"

DATA_ROOT="${PROJECT_DIR}/data/mvtec"
CACHE_SDVAE="${PROJECT_DIR}/cache/mvtec_sdvae"
CACHE_DINO="${PROJECT_DIR}/cache/mvtec_dinov2"
RUNS_DIR="${PROJECT_DIR}/runs"
RESULTS_DIR="${PROJECT_DIR}/results"

STAGE="${1:-}"
case "$STAGE" in
  cache_sdvae)
    python scripts/precompute_latents.py \
      --encoder sdvae \
      --data_root "$DATA_ROOT" \
      --output_dir "$CACHE_SDVAE" \
      --batch_size 64
    ;;
  cache_dinov2)
    python scripts/precompute_latents.py \
      --encoder dinov2 \
      --data_root "$DATA_ROOT" \
      --output_dir "$CACHE_DINO" \
      --batch_size 64
    ;;
  train_a0_uni)
    python train.py \
      --dataset mvtec --space pixel --data_root "$DATA_ROOT" \
      --no-class_cond --epochs 2000 --batch_size 32 --save_every 250 --num_samples 16 \
      --output_dir "${RUNS_DIR}/A0_pixel_uni" --resume
    ;;
  train_a0_cc)
    python train.py \
      --dataset mvtec --space pixel --data_root "$DATA_ROOT" \
      --class_cond --epochs 2000 --batch_size 32 --save_every 500 \
      --output_dir "${RUNS_DIR}/A0_pixel_cc" --resume
    ;;
  eval_a0_uni)
    CKPT="${2:-${RUNS_DIR}/A0_pixel_uni/ckpt_epoch2000.pt}"
    python evaluate.py \
      --dataset mvtec --data_root "$DATA_ROOT" --ckpt "$CKPT" \
      --output_dir "${RESULTS_DIR}/A0_pixel_uni" --save_vis
    ;;
  train_a1_vae)
    python train.py \
      --dataset mvtec --space vae --latent_cache "$CACHE_SDVAE" \
      --no-class_cond --epochs 1500 --batch_size 64 --save_every 250 \
      --output_dir "${RUNS_DIR}/A1_vae_uni" --resume
    ;;
  eval_a1_vae)
    CKPT="${2:-${RUNS_DIR}/A1_vae_uni/ckpt_epoch1500.pt}"
    python evaluate.py \
      --dataset mvtec --latent_cache "$CACHE_SDVAE" --ckpt "$CKPT" \
      --output_dir "${RESULTS_DIR}/A1_vae_uni_min" --aggregation min --save_vis
    python evaluate.py \
      --dataset mvtec --latent_cache "$CACHE_SDVAE" --ckpt "$CKPT" \
      --output_dir "${RESULTS_DIR}/A1_vae_uni_mean" --aggregation mean
    ;;
  *)
    echo "Usage: bash scripts/run_phase0_1.sh {cache_sdvae|cache_dinov2|train_a0_uni|train_a0_cc|eval_a0_uni|train_a1_vae|eval_a1_vae}"
    exit 1
    ;;
esac
