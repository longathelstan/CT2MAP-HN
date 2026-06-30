#!/usr/bin/env bash
# ============================================================================
# CT2MAP-HN: Full Evaluation Pipeline
# Chạy evaluation cho tất cả models: baseline, swin, distillation.
# Bao gồm: inference, metrics, MC Dropout uncertainty estimation.
#
# Usage:
#   bash scripts/eval_all.sh
#   bash scripts/eval_all.sh --models baseline swin distillation
#   bash scripts/eval_all.sh --split test --mc-passes 20
# ============================================================================
set -euo pipefail

# --- Defaults ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SPLIT="test"
MC_PASSES=20
MODELS=("baseline" "swin_unetr" "distillation")

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --split)     SPLIT="$2"; shift 2 ;;
        --mc-passes) MC_PASSES="$2"; shift 2 ;;
        --models)
            shift
            MODELS=()
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                MODELS+=("$1"); shift
            done
            ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo "=============================================="
echo " CT2MAP-HN: Full Evaluation Pipeline"
echo "=============================================="
echo " Split     : ${SPLIT}"
echo " MC Passes : ${MC_PASSES}"
echo " Models    : ${MODELS[*]}"
echo "=============================================="

cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# --- Config mapping ---
declare -A CONFIGS
CONFIGS["baseline"]="configs/baseline_nnunet.yaml"
CONFIGS["swin_unetr"]="configs/swin_unetr.yaml"
CONFIGS["distillation"]="configs/distill.yaml"

declare -A CHECKPOINTS
CHECKPOINTS["baseline"]="outputs/baseline/checkpoints/best_model.pt"
CHECKPOINTS["swin_unetr"]="outputs/swin_unetr/checkpoints/best_model.pt"
CHECKPOINTS["distillation"]="outputs/distillation/checkpoints/best_model.pt"

# --- Run evaluation for each model ---
FAILED=()

for model_name in "${MODELS[@]}"; do
    echo ""
    echo "----------------------------------------------"
    echo " Evaluating: ${model_name}"
    echo "----------------------------------------------"

    config="${CONFIGS[$model_name]:-}"
    checkpoint="${CHECKPOINTS[$model_name]:-}"

    if [ -z "$config" ] || [ ! -f "$config" ]; then
        echo "WARNING: Config not found for ${model_name}: ${config}"
        FAILED+=("${model_name}:config_missing")
        continue
    fi

    if [ -z "$checkpoint" ] || [ ! -f "$checkpoint" ]; then
        echo "WARNING: Checkpoint not found for ${model_name}: ${checkpoint}"
        FAILED+=("${model_name}:checkpoint_missing")
        continue
    fi

    output_dir="outputs/${model_name}/eval_${SPLIT}"
    mkdir -p "$output_dir"

    echo "  Config     : ${config}"
    echo "  Checkpoint : ${checkpoint}"
    echo "  Output     : ${output_dir}"

    # --- Step 1: Deterministic inference ---
    echo "  [1/3] Running inference..."
    python -m src.inference.infer_batch \
        --config "$config" \
        --checkpoint "$checkpoint" \
        --split "$SPLIT" \
        --output-dir "$output_dir/predictions" \
        2>&1 | tee "${output_dir}/inference.log" || {
        echo "ERROR: Inference failed for ${model_name}"
        FAILED+=("${model_name}:inference")
        continue
    }

    # --- Step 2: MC Dropout uncertainty ---
    echo "  [2/3] Running MC Dropout uncertainty (${MC_PASSES} passes)..."
    python -m src.inference.infer_batch \
        --config "$config" \
        --checkpoint "$checkpoint" \
        --split "$SPLIT" \
        --output-dir "$output_dir/uncertainty" \
        --mc-dropout \
        --mc-passes "$MC_PASSES" \
        2>&1 | tee "${output_dir}/uncertainty.log" || {
        echo "WARNING: MC Dropout failed for ${model_name} (non-fatal)"
        FAILED+=("${model_name}:mc_dropout")
    }

    # --- Step 3: Compute metrics ---
    echo "  [3/3] Computing metrics..."
    python -m src.inference.postprocess \
        --predictions-dir "$output_dir/predictions" \
        --output-dir "$output_dir/metrics" \
        --split "$SPLIT" \
        2>&1 | tee "${output_dir}/metrics.log" || {
        echo "WARNING: Metrics computation failed for ${model_name}"
        FAILED+=("${model_name}:metrics")
    }

    echo "  Done: ${model_name}"
done

# --- Summary ---
echo ""
echo "=============================================="
echo " Evaluation Summary"
echo "=============================================="
if [ ${#FAILED[@]} -eq 0 ]; then
    echo " All models evaluated successfully!"
else
    echo " Failures (${#FAILED[@]}):"
    for fail in "${FAILED[@]}"; do
        echo "   - ${fail}"
    done
fi
echo "=============================================="
