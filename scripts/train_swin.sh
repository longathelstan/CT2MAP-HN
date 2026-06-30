#!/usr/bin/env bash
# ============================================================================
# CT2MAP-HN: Swin UNETR Training (Student)
# Huấn luyện Swin UNETR student model, có thể chạy với Knowledge Distillation.
#
# Usage:
#   bash scripts/train_swin.sh
#   bash scripts/train_swin.sh --distill                  # with KD
#   bash scripts/train_swin.sh --config configs/swin_unetr.yaml --gpus 2
# ============================================================================
set -euo pipefail

# --- Defaults ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG="${PROJECT_ROOT}/configs/swin_unetr.yaml"
DISTILL_CONFIG="${PROJECT_ROOT}/configs/distill.yaml"
NUM_GPUS=4
MASTER_PORT=29501
USE_DISTILL=false

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)         CONFIG="$2"; shift 2 ;;
        --distill-config) DISTILL_CONFIG="$2"; shift 2 ;;
        --gpus)           NUM_GPUS="$2"; shift 2 ;;
        --port)           MASTER_PORT="$2"; shift 2 ;;
        --distill)        USE_DISTILL=true; shift ;;
        *)                echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo "=============================================="
echo " CT2MAP-HN: Swin UNETR Training"
echo "=============================================="
echo " Config    : ${CONFIG}"
echo " Distill   : ${USE_DISTILL}"
if [ "$USE_DISTILL" = true ]; then
    echo " KD Config : ${DISTILL_CONFIG}"
fi
echo " GPUs      : ${NUM_GPUS}"
echo " Port      : ${MASTER_PORT}"
echo "=============================================="

cd "$PROJECT_ROOT"

# --- Check configs ---
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config file not found: $CONFIG"
    exit 1
fi

# --- Set environment ---
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=$(seq -s',' 0 $((NUM_GPUS - 1)))

# --- Select training script ---
if [ "$USE_DISTILL" = true ]; then
    if [ ! -f "$DISTILL_CONFIG" ]; then
        echo "ERROR: Distillation config not found: $DISTILL_CONFIG"
        exit 1
    fi
    TRAIN_MODULE="src.train.train_distill"
    EXTRA_ARGS="--distill-config $DISTILL_CONFIG"
else
    TRAIN_MODULE="src.train.train_swin"
    EXTRA_ARGS=""
fi

# --- Launch training ---
if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Launching distributed training on ${NUM_GPUS} GPUs..."
    torchrun \
        --nproc_per_node="$NUM_GPUS" \
        --master_port="$MASTER_PORT" \
        -m "$TRAIN_MODULE" \
        --config "$CONFIG" \
        $EXTRA_ARGS
else
    echo "Launching single-GPU training..."
    python -m "$TRAIN_MODULE" --config "$CONFIG" $EXTRA_ARGS
fi

echo "=============================================="
echo " Swin UNETR training complete!"
echo "=============================================="
