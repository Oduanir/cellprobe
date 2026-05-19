#!/usr/bin/env bash
#
# Run the full data prep pipeline inside the BioNeMo container, on the AWS
# EC2 instance. Assumes the repo is at ~/bionemo-workspace/cellprobe and the
# BioNeMo cache is at ~/bionemo-cache.
#
# Usage:
#   ./scripts/run_data_pipeline.sh                  # all 3 diseases
#   ./scripts/run_data_pipeline.sh --only dcm       # one disease
#   ./scripts/run_data_pipeline.sh --force          # overwrite existing outputs
#
# The pipeline:
#   1. src.data.preprocess  → QC + stratified train/val/test split
#       → data/<disease>/splits/{train,val,test}/<disease>.h5ad
#   2. src.data.scdl_convert → BioNeMo memmap (rank-based gene tokenization)
#       → data/<disease>/scdl/{train,val,test}/
set -euo pipefail

BIONEMO_IMAGE="${BIONEMO_IMAGE:-nvcr.io/nvidia/clara/bionemo-framework:2.7.1}"
CACHE_DIR="${BIONEMO_CACHE:-$HOME/bionemo-cache}"
WORKSPACE="${CELLPROBE_DIR:-$HOME/bionemo-workspace/cellprobe}"

docker run --rm \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$CACHE_DIR:/root/.cache/bionemo" \
  -v "$WORKSPACE:/workspace/cellprobe" \
  -w /workspace/cellprobe \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e PYTHONUNBUFFERED=1 \
  "$BIONEMO_IMAGE" \
  bash -c "\
    python -u -m src.data.preprocess --config configs/diseases.yaml --out data/ $* && \
    python -u -m src.data.scdl_convert --config configs/diseases.yaml --out data/ $* \
  "
