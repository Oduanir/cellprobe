#!/usr/bin/env bash
#
# Frozen Geneformer + MLP probe pipeline.
# Mirrors NVIDIA's official benchmark methodology from
# `bionemo.geneformer.scripts.celltype_classification_bench` (linear probing
# on frozen pretrained embeddings, no fine-tuning).
#
# Two stages:
#   1. src.eval.extract_embeddings — GPU inference (Geneformer 10M) over
#      train/val/test SCDL data of each disease → predictions__rank_0.pt
#   2. src.eval.probe — sklearn MLPClassifier on frozen embeddings + labels
#
# Usage:
#   ./scripts/run_probe_pipeline.sh                  # all 3 diseases
#   ./scripts/run_probe_pipeline.sh --only uc        # one disease
#   ./scripts/run_probe_pipeline.sh --force          # overwrite existing
#   ./scripts/run_probe_pipeline.sh --probe-only     # skip extraction (re-run probe)
set -euo pipefail

BIONEMO_IMAGE="${BIONEMO_IMAGE:-nvcr.io/nvidia/clara/bionemo-framework:2.7.1}"
CACHE_DIR="${BIONEMO_CACHE:-$HOME/bionemo-cache}"
WORKSPACE="${CELLPROBE_DIR:-$HOME/bionemo-workspace/cellprobe}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-16}"

PROBE_ONLY=0
EXTRACT_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --probe-only) PROBE_ONLY=1 ;;
    *) EXTRACT_ARGS+=("$arg") ;;
  esac
done

# Common docker flags. Note: we mount the cache at /cache (not /root/.cache/bionemo)
# because /root is mode 700 in the container image and the non-root user
# can't walk through it even when the cache files themselves are accessible.
docker_run() {
  docker run --rm --gpus all \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "$CACHE_DIR:/cache" \
    -v "$WORKSPACE:/workspace/cellprobe" \
    -w /workspace/cellprobe \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp -e BIONEMO_CACHE=/cache -e PYTHONUNBUFFERED=1 \
    "$BIONEMO_IMAGE" "$@"
}

if [[ "$PROBE_ONLY" -eq 0 ]]; then
  echo "=== Stage 1: extract embeddings (GPU) ==="
  docker_run python -u -m src.eval.extract_embeddings \
    --micro-batch-size "$MICRO_BATCH_SIZE" \
    "${EXTRACT_ARGS[@]}"
fi

echo "=== Stage 2: MLP probe (CPU, sklearn) ==="
docker_run python -u -m src.eval.probe "${EXTRACT_ARGS[@]}"
