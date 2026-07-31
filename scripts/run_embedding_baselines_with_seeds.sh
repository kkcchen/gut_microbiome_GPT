#!/usr/bin/env bash
# Runs the embedding-baseline eval (extract embeddings from a pretrained checkpoint, then
# XGBoost/RandomForest/Linear/TabPFN on those embeddings) against the fixed split
# (/scratch/kchen13/hmc_final_fixed/downstream_{train,test}.h5ad), once per config.
#
# No outer seed loop: single-study tasks (no natural train/test boundary) are pooled and
# cross-validated by utils/downstream_utils.py::run_task_combined_cv, so the error bar comes
# from the CV folds themselves -- one run, one seed, following kd_tasks/run_downstream_final.py
# (which never reruns across seeds either). Multi-study tasks (location, sex -- disjoint
# train/test studies, a real cross-study holdout) keep the original presplit evaluation, also
# one run -- for those, each method's own hyperparameter search (search_type: grid/random)
# reports its inner-CV std at the winning params as a proxy margin, written to
# <task>/<method>/classification/search_cv_std.json (see utils/downstream_utils.py::
# run_single_method), the same trick kd_tasks/downstream_final_utils.py::run_presplit uses.
# See utils/downstream_split_mode.py::resolve_split_mode for the mode detection and
# downstream_tasks_config.cv in each embed config for the CV settings (outer_folds, default 5).
#
# Usage:
#   scripts/run_embedding_baselines_with_seeds.sh [OUTPUT_ROOT]
#
# Env overrides:
#   EMBED_CONFIGS       default: the two winning checkpoints' embedding_baselines configs
#   MAIN_PROCESS_PORT   default: 0 (passed to `accelerate launch`)

set -euo pipefail

OUTPUT_ROOT="${1:-outputs/eval/embedding_baselines_seeds}"

EMBED_CONFIGS="${EMBED_CONFIGS:-configs/eval/embedding_baselines/pretrained_baseline_bins20_mask030.yaml configs/eval/embedding_baselines/stage3_abundance_emb_concatenation.yaml}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-0}"

echo "Running embedding baselines (pooled CV for single-study tasks, presplit for the rest)"
echo "  Split (fixed): /scratch/kchen13/hmc_final_fixed/downstream_{train,test}.h5ad"
echo "  Results -> ${OUTPUT_ROOT}/<config_name>"

for embed_cfg in ${EMBED_CONFIGS}; do
    cfg_name="$(basename "${embed_cfg}" .yaml)"
    run_output_dir="${OUTPUT_ROOT}/${cfg_name}"

    echo ""
    echo "=== checkpoint=${cfg_name}: extracting embeddings + baselines ==="
    accelerate launch --main_process_port "${MAIN_PROCESS_PORT}" -m scripts.eval \
        --config "${embed_cfg}" \
        paths.output_dir="${run_output_dir}"
done

echo ""
echo "Done. See ${OUTPUT_ROOT}/combined_summary.md (per-task means; cv_summary.json under each"
echo "task/method for pooled tasks has the full per-fold spread, search_cv_std.json under each"
echo "presplit task/method has the inner-search proxy margin)."
