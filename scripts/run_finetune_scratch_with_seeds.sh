#!/usr/bin/env bash
# Reruns the "finetuned from scratch" (no-pretraining) baselines -- max_epochs=0 random-init
# encoder, immediately followed by the auto-finetune-all-tasks chain -- across several
# different stratified train/test-split seeds, so that
# utils.finetune_orchestration.combine_seeded_finetune_summaries can compute mean +/- std
# ("error bars") per task/config. Each seed varies both the data split AND the model's
# random weight initialization (training.seed), since both are stochastic components of
# a "from scratch" run.
#
# Unlike scripts/run_finetune_pretrained_with_seeds.sh, this reruns scripts/train.py itself
# (unmodified) with a fresh paths.output_dir per seed -- there's no expensive pretraining to
# avoid repeating here (max_epochs=0), so no separate re-finetune-only script is needed.
#
# Usage:
#   scripts/run_finetune_scratch_with_seeds.sh [N_SEEDS] [SPLIT_ROOT] [OUTPUT_ROOT]
#
# Env overrides:
#   PREPROCESS_CONFIG   default: configs/preprocessing_and_split.yaml
#   SCRATCH_CONFIGS      default: the two baseline_random_init_*.yaml configs
#   START_SEED           default: 0
#   TRAINING_SEED_BASE    default: 1000 (offset added to the loop index for training.seed,
#                          kept well clear of the split seeds passed to scripts.preprocess)

set -euo pipefail

N_SEEDS="${1:-5}"
SPLIT_ROOT="${2:-data/seeded_splits/finetune_scratch}"
OUTPUT_ROOT="${3:-outputs/pretrain/real_runs/baseline_seeds}"

PREPROCESS_CONFIG="${PREPROCESS_CONFIG:-configs/preprocessing_and_split.yaml}"
SCRATCH_CONFIGS="${SCRATCH_CONFIGS:-configs/pretrain/real_runs/baseline_random_init_default_scgpt.yaml configs/pretrain/real_runs/baseline_random_init_winner_arch.yaml}"
START_SEED="${START_SEED:-0}"
TRAINING_SEED_BASE="${TRAINING_SEED_BASE:-1000}"

echo "Running from-scratch finetune baselines across ${N_SEEDS} seeds (starting at ${START_SEED})"
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

    for scratch_cfg in ${SCRATCH_CONFIGS}; do
        cfg_name="$(basename "${scratch_cfg}" .yaml)"
        run_output_dir="${OUTPUT_ROOT}/seed_${i}/${cfg_name}"

        echo ""
        echo "=== Seed ${i}, config=${cfg_name}: random-init + finetune on all tasks ==="
        python -m scripts.train \
            --config "${scratch_cfg}" \
            paths.output_dir="${run_output_dir}" \
            training.seed="$((TRAINING_SEED_BASE + i))" \
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
