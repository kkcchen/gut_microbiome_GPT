#!/usr/bin/env bash
# Reruns the embedding-baseline eval (extract embeddings from a pretrained checkpoint, then
# XGBoost/RandomForest/Linear/TabPFN on those embeddings) across several different stratified
# train/test-split seeds, so that scripts/aggregate_seeded_downstream_summary.py can compute
# mean +/- std ("error bars") per task/method/config. Does not repeat pretraining -- it reuses
# the existing checkpoint and only re-extracts embeddings + retrains the classical baselines
# for each split.
#
# Usage:
#   scripts/run_embedding_baselines_with_seeds.sh [N_SEEDS] [SPLIT_ROOT] [OUTPUT_ROOT]
#
# Env overrides:
#   PREPROCESS_CONFIG   default: configs/preprocessing_and_split.yaml
#   EMBED_CONFIGS        default: the two winning checkpoints' embedding_baselines configs
#   START_SEED           default: 0
#   MAIN_PROCESS_PORT     default: 0 (passed to `accelerate launch`)

set -euo pipefail

N_SEEDS="${1:-5}"
SPLIT_ROOT="${2:-data/seeded_splits/embedding_baselines}"
OUTPUT_ROOT="${3:-outputs/eval/embedding_baselines_seeds}"

PREPROCESS_CONFIG="${PREPROCESS_CONFIG:-configs/preprocessing_and_split.yaml}"
EMBED_CONFIGS="${EMBED_CONFIGS:-configs/eval/embedding_baselines/pretrained_baseline_bins20_mask030.yaml configs/eval/embedding_baselines/stage3_abundance_emb_concatenation.yaml}"
START_SEED="${START_SEED:-0}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-0}"

echo "Running embedding baselines across ${N_SEEDS} seeds (starting at ${START_SEED})"
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

    for embed_cfg in ${EMBED_CONFIGS}; do
        cfg_name="$(basename "${embed_cfg}" .yaml)"
        run_output_dir="${OUTPUT_ROOT}/seed_${i}/${cfg_name}"

        echo ""
        echo "=== Seed ${i}, checkpoint=${cfg_name}: extracting embeddings + baselines ==="
        accelerate launch --main_process_port "${MAIN_PROCESS_PORT}" -m scripts.eval \
            --config "${embed_cfg}" \
            paths.output_dir="${run_output_dir}" \
            "paths.eval_files=[${seed_dir}/downstream_train.h5ad,${seed_dir}/downstream_test.h5ad]"
    done
done

echo ""
echo "=== Aggregating across seeds ==="
python -m scripts.aggregate_seeded_downstream_summary --seeds-root "${OUTPUT_ROOT}"

echo ""
echo "Done. See ${OUTPUT_ROOT}/combined_summary_with_error_bars.md"
