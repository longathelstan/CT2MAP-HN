#!/bin/bash
cd /data/lowngworkspace/CT2MAP-HN
export OMP_NUM_THREADS=1 KMP_AFFINITY=disabled MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "=== prepare_data REMAINING (16 workers, overwrite) ==="; date
python3 scripts/prepare_data.py --config configs/preprocess.yaml --manifest data/processed/manifests/manifest_remaining.csv --workers 16 --overwrite
echo "=== create_targets ALL 782 (16 workers, overwrite) ==="; date
python3 scripts/create_targets.py --config configs/data.yaml --workers 16 --overwrite
echo "=== ALL_DONE ==="; date
