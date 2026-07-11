#!/usr/bin/env bash
# ============================================================================
# CT2MAP-HN: Baseline Model Training
# Huấn luyện baseline model (MONAI BasicUNet) cho metabolic heatmap regression.
#
# Usage:
#   bash scripts/train_baseline.sh
#   bash scripts/train_baseline.sh --gpus 2
#   bash scripts/train_baseline.sh --config configs/baseline_nnunet.yaml
# ============================================================================
set -euo pipefail

# --- Defaults ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG="${PROJECT_ROOT}/configs/baseline_p40.yaml"
NUM_GPUS=2
MASTER_PORT=29500

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --gpus)   NUM_GPUS="$2"; shift 2 ;;
        --port)   MASTER_PORT="$2"; shift 2 ;;
        *)        echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo "=============================================="
echo " CT2MAP-HN: Baseline Training"
echo "=============================================="
echo " Config    : ${CONFIG}"
echo " GPUs      : ${NUM_GPUS}"
echo " Port      : ${MASTER_PORT}"
echo "=============================================="

cd "$PROJECT_ROOT"

# --- Check config exists ---
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config file not found: $CONFIG"
    exit 1
fi

# --- Set environment ---
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=$(seq -s',' 0 $((NUM_GPUS - 1)))

# --- Launch training ---
if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Launching distributed training on ${NUM_GPUS} GPUs..."
    torchrun \
        --nproc_per_node="$NUM_GPUS" \
        --master_port="$MASTER_PORT" \
        -m src.train.train_baseline \
        --config "$CONFIG"
else
    echo "Launching single-GPU training..."
    python -m src.train.train_baseline --config "$CONFIG"
fi

echo "=============================================="
echo " Baseline training complete!"
echo "=============================================="
