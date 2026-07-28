#!/usr/bin/env bash
# Reruns the raw-baseline (XGBoost/Linear on raw taxa tables) downstream eval across
# several different stratified train/test-split seeds, so that
# scripts/aggregate_seeded_downstream_summary.py can compute mean +/- std ("error bars")
# per task/method/normalization. The pipeline otherwise only ever evaluates on one fixed
# split, so this is what stands in for k-fold cross-validation here.
#
# Usage:
#   scripts/run_raw_baselines_with_seeds.sh [N_SEEDS] [SPLIT_ROOT] [OUTPUT_ROOT]
#
# Env overrides:
#   PREPROCESS_CONFIG   default: configs/preprocessing_and_split.yaml
#   RAW_EVAL_CONFIG      default: configs/eval/raw.yaml
#   NORMALIZATIONS       default: "clr log_rel_abundance none rel_ab"
#   START_SEED           default: 0

set -euo pipefail

N_SEEDS="${1:-5}"
SPLIT_ROOT="${2:-data/seeded_splits/raw_baselines}"
OUTPUT_ROOT="${3:-outputs/eval/raw_baselines_seeds}"

PREPROCESS_CONFIG="${PREPROCESS_CONFIG:-configs/preprocessing_and_split.yaml}"
RAW_EVAL_CONFIG="${RAW_EVAL_CONFIG:-configs/eval/raw.yaml}"
# NOTE: the currently-implemented methods in utils/downstream_data_utils.py::normalize_embeddings
# are "clr", "log" (log1p of raw counts), "none", "rel_ab", "l2" -- there is no literal
# "log_rel_abundance" method despite that being a config name in outputs/eval/raw_baselines/, so
# this defaults to "log" as the closest current equivalent. Check that against what you actually
# want before trusting the log-normalization row in the aggregated output.
NORMALIZATIONS="${NORMALIZATIONS:-clr log none rel_ab}"
START_SEED="${START_SEED:-0}"

echo "Running raw baselines across ${N_SEEDS} seeds (starting at ${START_SEED})"
echo "  Splits  -> ${SPLIT_ROOT}/seed_<i>"
echo "  Results -> ${OUTPUT_ROOT}/seed_<i>/<normalization>"

for ((i = START_SEED; i < START_SEED + N_SEEDS; i++)); do
    seed_dir="${SPLIT_ROOT}/seed_${i}"

    echo ""
    echo "=== Seed ${i}: preprocessing (split seed=${i}) ==="
    python -m scripts.preprocess \
        --config "${PREPROCESS_CONFIG}" \
        --seed "${i}" \
        --save-path "${seed_dir}"

    for norm in ${NORMALIZATIONS}; do
        run_output_dir="${OUTPUT_ROOT}/seed_${i}/${norm}"

        echo ""
        echo "=== Seed ${i}, normalization=${norm}: xgboost + linear ==="
        python -m scripts.eval_downstream_only \
            --config "${RAW_EVAL_CONFIG}" \
            paths.output_dir="${run_output_dir}" \
            paths.train_path="${seed_dir}/downstream_train.h5ad" \
            paths.test_path="${seed_dir}/downstream_test.h5ad" \
            downstream_tasks_config.normalization="${norm}" \
            downstream_tasks_config.methods.linear.search_type=none
    done
done

echo ""
echo "=== Aggregating across seeds ==="
python -m scripts.aggregate_seeded_downstream_summary --seeds-root "${OUTPUT_ROOT}"

echo ""
echo "Done. See ${OUTPUT_ROOT}/combined_summary_with_error_bars.md"
