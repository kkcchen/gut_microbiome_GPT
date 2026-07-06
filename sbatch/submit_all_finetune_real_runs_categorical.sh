#!/bin/bash
# Submits sbatch/finetune_real_runs_categorical.sbatch for every pretrain run config
# in configs/pretrain/real_runs/.
#
# Usage: ./sbatch/submit_all_finetune_real_runs_categorical.sh

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

CONFIG_DIR="configs/pretrain/real_runs"

RUN_NAMES=()
for CONFIG in "${CONFIG_DIR}"/*.yaml; do
    RUN_NAMES+=("$(basename "${CONFIG}" .yaml)")
done

for RUN_NAME in "${RUN_NAMES[@]}"; do
    echo "Submitting finetune sweep for run: ${RUN_NAME}"
    sbatch sbatch/finetune_real_runs_categorical.sbatch "${RUN_NAME}"
done
