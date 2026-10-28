#!/usr/bin/env bash
# Local paths only; no training is started by sourcing this file.
vladbuff_project="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export GSV_CITIES_ROOT="${GSV_CITIES_ROOT:-/home/admin123/github/WORKSPACE/dataset/gsv-cities}"
export PITTSBURGH_ROOT="${PITTSBURGH_ROOT:-$vladbuff_project/datasets/Pittsburgh/Pittsburgh}"
export MSLS_ROOT="${MSLS_ROOT:-$vladbuff_project/logs/active_dataset_views/MSLS}"
if [[ -d "$vladbuff_project/logs/active_dataset_manifests" ]]; then
  export VLADBUFF_MANIFEST_DIR="${VLADBUFF_MANIFEST_DIR:-$vladbuff_project/logs/active_dataset_manifests}"
fi
export DINOV2_REPO="${DINOV2_REPO:-/home/admin123/.cache/torch/hub/facebookresearch_dinov2_main}"
export VLADBUFF_PYTHON="${VLADBUFF_PYTHON:-/home/admin123/miniconda3/envs/vladbuff/bin/python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/vladbuff-matplotlib}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
