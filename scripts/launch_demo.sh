#!/usr/bin/env bash
# ============================================================================
# CT2MAP-HN: Launch Streamlit Demo Dashboard
# Khởi chạy demo tương tác để visualize metabolic risk heatmap.
#
# Usage:
#   bash scripts/launch_demo.sh
#   bash scripts/launch_demo.sh --config configs/demo.yaml
#   bash scripts/launch_demo.sh --port 8502
# ============================================================================
set -euo pipefail

# --- Defaults ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG="${PROJECT_ROOT}/configs/demo.yaml"
PORT=8501
HOST="0.0.0.0"

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --port)   PORT="$2"; shift 2 ;;
        --host)   HOST="$2"; shift 2 ;;
        *)        echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo "=============================================="
echo " CT2MAP-HN: Streamlit Demo"
echo "=============================================="
echo " Config    : ${CONFIG}"
echo " Address   : http://${HOST}:${PORT}"
echo "=============================================="

cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# --- Check Streamlit is installed ---
if ! command -v streamlit &> /dev/null; then
    echo "ERROR: Streamlit is not installed."
    echo "  Install with: pip install streamlit"
    exit 1
fi

# --- Check config exists ---
if [ ! -f "$CONFIG" ]; then
    echo "WARNING: Config file not found: $CONFIG"
    echo "  Running with default settings."
fi

# --- Launch ---
echo "Starting Streamlit demo..."
streamlit run src/demo/app_streamlit.py \
    --server.port "$PORT" \
    --server.address "$HOST" \
    --server.maxUploadSize 500 \
    --browser.gatherUsageStats false \
    -- --config "$CONFIG"
