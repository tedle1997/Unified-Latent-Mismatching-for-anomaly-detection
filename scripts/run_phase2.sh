#!/usr/bin/env bash
# Runner script for Phase 2 (Setup B: Unified Latents from scratch + Flow Mismatching).
# Usage: bash scripts/run_phase2.sh <stage> [ckpt_path]
# Stages: train_b1 | eval_b1 | train_b_lo | eval_b_lo | train_b_hi | eval_b_hi | diag_b1_ood
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
RUNS_DIR="${PROJECT_DIR}/runs"
RESULTS_DIR="${PROJECT_DIR}/results"

STAGE="${1:-}"
case "$STAGE" in
  train_b1)
    # Setup B1: sigma0 = 0.08, 32x32x8 latents, label-free (--no-class_cond), lean 5.2M train-only decoder.
    python train.py \
      --dataset mvtec --space ul --data_root "$DATA_ROOT" \
      --no-class_cond --ul_sigma0 0.08 --ul_channels 8 --ul_downsample 4 \
      --ul_dec_channels 32 --ul_decoder_bias 4.0 --ul_loss_factor 2.5 --ul_prior_warmup_epochs 5 \
      --epochs 600 --batch_size 32 --save_every 50 --num_samples 16 --ode_steps 50 \
      --output_dir "${RUNS_DIR}/B1_ul_s008" --resume
    ;;
  eval_b1)
    CKPT="${2:-${RUNS_DIR}/B1_ul_s008/ckpt_epoch0600.pt}"
    # 1. Flow Mismatching (min) with sigma0-aligned t_range
    python evaluate.py --dataset mvtec --data_root "$DATA_ROOT" --ckpt "$CKPT" \
      --aggregation min --save_vis --output_dir "${RESULTS_DIR}/B1_ul_s008_min_aligned"
    # 2. Raw UL Prior ELBO (mean) with sigma0-aligned t_range
    python evaluate.py --dataset mvtec --data_root "$DATA_ROOT" --ckpt "$CKPT" \
      --aggregation mean --output_dir "${RESULTS_DIR}/B1_ul_s008_mean_aligned"
    # 3. Unaligned t_range=(0, 1) ablation
    python evaluate.py --dataset mvtec --data_root "$DATA_ROOT" --ckpt "$CKPT" \
      --aggregation min --t1 1.0 --output_dir "${RESULTS_DIR}/B1_ul_s008_min_t1"
    ;;
  diag_b1_ood)
    CKPT="${2:-${RUNS_DIR}/B1_ul_s008/ckpt_epoch0600.pt}"
    python evaluate.py --dataset mvtec --data_root "$DATA_ROOT" --ckpt "$CKPT" \
      --aggregation min --diag_recon --diag_recon_steps 20 --diag_recon_max_per_class 20 \
      --output_dir "${RESULTS_DIR}/B1_ul_s008_ood_diag"
    ;;
  train_b_lo)
    # High bitrate (low noise): sigma0 = 0.04
    python train.py \
      --dataset mvtec --space ul --data_root "$DATA_ROOT" \
      --no-class_cond --ul_sigma0 0.04 --ul_channels 8 --ul_downsample 4 \
      --ul_dec_channels 32 --ul_decoder_bias 4.0 --ul_loss_factor 2.5 --ul_prior_warmup_epochs 5 \
      --epochs 600 --batch_size 32 --save_every 100 --num_samples 16 --ode_steps 50 \
      --output_dir "${RUNS_DIR}/B_lo_ul_s004" --resume
    ;;
  eval_b_lo)
    CKPT="${2:-${RUNS_DIR}/B_lo_ul_s004/ckpt_epoch0600.pt}"
    python evaluate.py --dataset mvtec --data_root "$DATA_ROOT" --ckpt "$CKPT" \
      --aggregation min --output_dir "${RESULTS_DIR}/B_lo_ul_s004_min"
    python evaluate.py --dataset mvtec --data_root "$DATA_ROOT" --ckpt "$CKPT" \
      --aggregation mean --no-diag_bitrate --output_dir "${RESULTS_DIR}/B_lo_ul_s004_mean"
    python evaluate.py --dataset mvtec --data_root "$DATA_ROOT" --ckpt "$CKPT" \
      --aggregation min --t1 1.0 --no-diag_bitrate --output_dir "${RESULTS_DIR}/B_lo_ul_s004_min_t1"
    ;;
  train_b_hi)
    # Low bitrate (high noise): sigma0 = 0.16
    python train.py \
      --dataset mvtec --space ul --data_root "$DATA_ROOT" \
      --no-class_cond --ul_sigma0 0.16 --ul_channels 8 --ul_downsample 4 \
      --ul_dec_channels 32 --ul_decoder_bias 4.0 --ul_loss_factor 2.5 --ul_prior_warmup_epochs 5 \
      --epochs 600 --batch_size 32 --save_every 100 --num_samples 16 --ode_steps 50 \
      --output_dir "${RUNS_DIR}/B_hi_ul_s016" --resume
    ;;
  eval_b_hi)
    CKPT="${2:-${RUNS_DIR}/B_hi_ul_s016/ckpt_epoch0600.pt}"
    python evaluate.py --dataset mvtec --data_root "$DATA_ROOT" --ckpt "$CKPT" \
      --aggregation min --output_dir "${RESULTS_DIR}/B_hi_ul_s016_min"
    python evaluate.py --dataset mvtec --data_root "$DATA_ROOT" --ckpt "$CKPT" \
      --aggregation mean --no-diag_bitrate --output_dir "${RESULTS_DIR}/B_hi_ul_s016_mean"
    python evaluate.py --dataset mvtec --data_root "$DATA_ROOT" --ckpt "$CKPT" \
      --aggregation min --t1 1.0 --no-diag_bitrate --output_dir "${RESULTS_DIR}/B_hi_ul_s016_min_t1"
    ;;
  cache_b1_stage2)
    CKPT="${2:-${RUNS_DIR}/B1_ul_s008/ckpt_epoch0600.pt}"
    python scripts/precompute_latents.py \
      --encoder ul_ckpt --ul_ckpt "$CKPT" --dataset mvtec \
      --data_root "$DATA_ROOT" --output_dir "${PROJECT_DIR}/cache/mvtec_b1_ul_s008" \
      --batch_size 64
    ;;
  train_b1_stage2)
    # UL Stage 2 (arXiv:2602.17270 Sec. 3.3): freeze Stage-1 encoder E_psi, train clean uniform-MSE
    # Flow Matching base model on standardized 32x32x8 UL latents (no decoder, no ELBO skew).
    python train.py \
      --dataset mvtec --space ul_stage2 --latent_cache "${PROJECT_DIR}/cache/mvtec_b1_ul_s008" \
      --no-class_cond --epochs 1500 --batch_size 64 --save_every 250 --num_samples 0 \
      --output_dir "${RUNS_DIR}/B1_ul_s008_stage2" --resume
    ;;
  eval_b1_stage2)
    CKPT="${2:-${RUNS_DIR}/B1_ul_s008_stage2/ckpt_epoch1500.pt}"
    # Cached eval (min vs mean) + online eval with frozen Stage-1 encoder for exact end-to-end latency.
    python evaluate.py --dataset mvtec --latent_cache "${PROJECT_DIR}/cache/mvtec_b1_ul_s008" --ckpt "$CKPT" \
      --aggregation min --num_paths 10 --num_steps 10 --save_vis \
      --output_dir "${RESULTS_DIR}/B1_ul_s008_stage2_min"
    python evaluate.py --dataset mvtec --latent_cache "${PROJECT_DIR}/cache/mvtec_b1_ul_s008" --ckpt "$CKPT" \
      --aggregation mean --num_paths 10 --num_steps 10 \
      --output_dir "${RESULTS_DIR}/B1_ul_s008_stage2_mean"
    ;;
  *)
    echo "Usage: bash scripts/run_phase2.sh {train_b1|eval_b1|diag_b1_ood|cache_b1_stage2|train_b1_stage2|eval_b1_stage2|train_b_lo|eval_b_lo|train_b_hi|eval_b_hi} [ckpt]"
    exit 1
    ;;
esac
