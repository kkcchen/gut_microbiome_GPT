#!/usr/bin/env bash
# Re-finetunes one or more ALREADY-TRAINED pretrain checkpoints on all registered
# downstream tasks, across several different stratified train/test-split seeds, so that
# utils.finetune_orchestration.combine_seeded_finetune_summaries can compute mean +/- std
# ("error bars") per task/config. Does NOT repeat pretraining -- it reuses the existing
# checkpoint/vocab via scripts/finetune_all_tasks.py and only varies the finetuning data split.
#
# Usage:
#   scripts/run_finetune_pretrained_with_seeds.sh [N_SEEDS] [SPLIT_ROOT] [OUTPUT_ROOT]
#
# Env overrides:
#   PREPROCESS_CONFIG   default: configs/preprocessing_and_split.yaml
#   PRETRAIN_CONFIGS     default: the two winning checkpoints' pretrain configs
#   START_SEED           default: 0

set -euo pipefail

N_SEEDS="${1:-5}"
SPLIT_ROOT="${2:-data/seeded_splits/finetune_pretrained}"
OUTPUT_ROOT="${3:-outputs/pretrain/finetune_pretrained_seeds}"

PREPROCESS_CONFIG="${PREPROCESS_CONFIG:-configs/preprocessing_and_split.yaml}"
PRETRAIN_CONFIGS="${PRETRAIN_CONFIGS:-configs/pretrain/pretrained_baseline/pretrained_baseline_bins20_mask030.yaml configs/pretrain/real_runs/stage3_abundance_emb_concatenation.yaml}"
START_SEED="${START_SEED:-0}"

echo "Re-finetuning ${PRETRAIN_CONFIGS} across ${N_SEEDS} seeds (starting at ${START_SEED})"
echo "  Splits  -> ${SPLIT_ROOT}/seed_<i>"
echo "  Results -> ${OUTPUT_ROOT}/seed_<i>/<config_name>"

for ((i = START_SEED; i < START_SEED + N_SEEDS; i++)); do
    seed_dir="${SPLIT_ROOT}/seed_${i}"

    echo ""
    echo "=== Seed ${i}: preprocessing (split seed=${i}) ==="
    python -m scripts.preprocess \
        --config "${PREPROCESS_CONFIG}" \
        --seed "${i}" \
        --save-path "${seed_dir}"

    for pretrain_cfg in ${PRETRAIN_CONFIGS}; do
        cfg_name="$(basename "${pretrain_cfg}" .yaml)"
        run_output_root="${OUTPUT_ROOT}/seed_${i}/${cfg_name}"

        echo ""
        echo "=== Seed ${i}, checkpoint=${cfg_name}: re-finetuning on all tasks ==="
        python -m scripts.finetune_all_tasks \
            --config "${pretrain_cfg}" \
            --finetune-output-root "${run_output_root}" \
            finetune.paths.downstream_train="${seed_dir}/downstream_train.h5ad" \
            finetune.paths.downstream_test="${seed_dir}/downstream_test.h5ad"
    done
done

echo ""
echo "=== Aggregating across seeds ==="
python -c "
from utils.finetune_orchestration import combine_seeded_finetune_summaries
combine_seeded_finetune_summaries('${OUTPUT_ROOT}')
"

echo ""
echo "Done. See ${OUTPUT_ROOT}/combined_summary_with_error_bars.md"
