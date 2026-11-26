#!/usr/bin/env bash
# Runner script for Phase 3 (Setup C: Frozen DINOv2 + UL Adapter + Flow Mismatching, and MVTec-LOCO).
# Usage: bash scripts/run_phase3.sh <stage> [ckpt_path]
# Stages: train_c1 | eval_c1 | diag_c1_ood | train_c_dim32 | train_c_dim64
#         download_loco | cache_loco_dinov2 | train_loco_c1 | eval_loco_c1
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

DATA_MVTEC="${PROJECT_DIR}/data/mvtec"
DATA_LOCO="${PROJECT_DIR}/data/mvtec_loco"
CACHE_MVTEC_DINO="${PROJECT_DIR}/cache/mvtec_dinov2"
CACHE_LOCO_DINO="${PROJECT_DIR}/cache/mvtec_loco_dinov2"
RUNS_DIR="${PROJECT_DIR}/runs"
RESULTS_DIR="${PROJECT_DIR}/results"

STAGE="${1:-}"
case "$STAGE" in
  train_c1)
    # Setup C1: Frozen DINOv2 ViT-S/14 (16x16x384) + MLP UL adapter -> 16x16x16, sigma0=0.08, label-free.
    python train.py \
      --dataset mvtec --space ul_dino --latent_cache "$CACHE_MVTEC_DINO" \
      --no-class_cond --ul_sigma0 0.08 --ul_channels 16 --ul_adapter_type mlp --ul_adapter_hidden 256 \
      --ul_dec_channels 32 --ul_decoder_bias 4.0 --ul_loss_factor 2.5 --ul_prior_warmup_epochs 5 \
      --epochs 600 --batch_size 32 --save_every 100 --num_samples 16 --ode_steps 50 \
      --output_dir "${RUNS_DIR}/C1_ul_dino_c16_s008" --resume
    ;;
  eval_c1)
    CKPT="${2:-${RUNS_DIR}/C1_ul_dino_c16_s008/ckpt_epoch0600.pt}"
    python evaluate.py --dataset mvtec --latent_cache "$CACHE_MVTEC_DINO" --ckpt "$CKPT" \
      --aggregation min --save_vis --output_dir "${RESULTS_DIR}/C1_ul_dino_c16_min"
    python evaluate.py --dataset mvtec --latent_cache "$CACHE_MVTEC_DINO" --ckpt "$CKPT" \
      --aggregation mean --output_dir "${RESULTS_DIR}/C1_ul_dino_c16_mean"
    python evaluate.py --dataset mvtec --latent_cache "$CACHE_MVTEC_DINO" --ckpt "$CKPT" \
      --aggregation min --t1 1.0 --output_dir "${RESULTS_DIR}/C1_ul_dino_c16_min_t1"
    ;;
  diag_c1_ood)
    CKPT="${2:-${RUNS_DIR}/C1_ul_dino_c16_s008/ckpt_epoch0600.pt}"
    python evaluate.py --dataset mvtec --latent_cache "$CACHE_MVTEC_DINO" --ckpt "$CKPT" \
      --aggregation min --diag_recon --diag_recon_steps 20 --diag_recon_max_per_class 20 \
      --output_dir "${RESULTS_DIR}/C1_ul_dino_c16_ood_diag"
    ;;
  train_c_dim32)
    python train.py \
      --dataset mvtec --space ul_dino --latent_cache "$CACHE_MVTEC_DINO" \
      --no-class_cond --ul_sigma0 0.08 --ul_channels 32 --ul_adapter_type mlp --ul_adapter_hidden 256 \
      --ul_dec_channels 32 --ul_decoder_bias 4.0 --ul_loss_factor 2.5 --ul_prior_warmup_epochs 5 \
      --epochs 600 --batch_size 32 --save_every 100 --num_samples 16 --ode_steps 50 \
      --output_dir "${RUNS_DIR}/C_dim32_ul_dino_s008" --resume
    ;;
  train_c_dim64)
    python train.py \
      --dataset mvtec --space ul_dino --latent_cache "$CACHE_MVTEC_DINO" \
      --no-class_cond --ul_sigma0 0.08 --ul_channels 64 --ul_adapter_type mlp --ul_adapter_hidden 256 \
      --ul_dec_channels 32 --ul_decoder_bias 4.0 --ul_loss_factor 2.5 --ul_prior_warmup_epochs 5 \
      --epochs 600 --batch_size 32 --save_every 100 --num_samples 16 --ode_steps 50 \
      --output_dir "${RUNS_DIR}/C_dim64_ul_dino_s008" --resume
    ;;
  download_loco)
    python -c "
from huggingface_hub import snapshot_download
from pathlib import Path
out = Path('${DATA_LOCO}')
out.mkdir(parents=True, exist_ok=True)
snapshot_download(repo_id='Rafa612/mvtec_loco', repo_type='dataset', local_dir=str(out), ignore_patterns=['*.DS_Store', '.gitattributes'], max_workers=8)
print('MVTec-LOCO downloaded to', out)
"
    ;;
  cache_loco_dinov2)
    python scripts/precompute_latents.py \
      --encoder dinov2 --data_root "$DATA_LOCO" --output_dir "$CACHE_LOCO_DINO" --batch_size 64
    ;;
  train_loco_c1)
    python train.py \
      --dataset mvtec_loco --space ul_dino --latent_cache "$CACHE_LOCO_DINO" \
      --no-class_cond --ul_sigma0 0.08 --ul_channels 16 --ul_adapter_type mlp --ul_adapter_hidden 256 \
      --ul_dec_channels 32 --ul_decoder_bias 4.0 --ul_loss_factor 2.5 --ul_prior_warmup_epochs 5 \
      --epochs 600 --batch_size 32 --save_every 100 --num_samples 16 --ode_steps 50 \
      --output_dir "${RUNS_DIR}/LOCO_C1_ul_dino_c16_s008" --resume
    ;;
  eval_loco_c1)
    CKPT="${2:-${RUNS_DIR}/LOCO_C1_ul_dino_c16_s008/ckpt_epoch0600.pt}"
    python evaluate.py --dataset mvtec_loco --latent_cache "$CACHE_LOCO_DINO" --ckpt "$CKPT" \
      --aggregation min --save_vis --output_dir "${RESULTS_DIR}/LOCO_C1_ul_dino_min"
    python evaluate.py --dataset mvtec_loco --latent_cache "$CACHE_LOCO_DINO" --ckpt "$CKPT" \
      --aggregation mean --output_dir "${RESULTS_DIR}/LOCO_C1_ul_dino_mean"
    ;;
  *)
    echo "Usage: bash scripts/run_phase3.sh {train_c1|eval_c1|diag_c1_ood|train_c_dim32|train_c_dim64|download_loco|cache_loco_dinov2|train_loco_c1|eval_loco_c1} [ckpt]"
    exit 1
    ;;
esac
