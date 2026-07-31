#!/usr/bin/env bash
# Re-finetunes one or more ALREADY-TRAINED pretrain checkpoints on all registered
# downstream tasks, once per checkpoint, against the fixed split
# (/scratch/kchen13/hmc_final_fixed/downstream_{train,test}.h5ad). Does NOT repeat
# pretraining -- it reuses the existing checkpoint/vocab via scripts/finetune_all_tasks.py.
#
# Two passes per checkpoint:
#   1) One full run, no seed loop: single-study tasks (no natural train/test boundary) are
#      pooled and K-fold cross-validated by utils/finetune_cv_utils.py::run_task_cv (one
#      training run per fold), so the error bar comes from the CV folds themselves -- one
#      run, one seed is enough. Multi-study tasks (location, sex -- disjoint train/test
#      studies, a real cross-study holdout) also run once here, giving the usual single
#      point estimate. See utils/downstream_split_mode.py and configs/finetune/task_registry.yaml
#      for the mode detection and CV settings (outer_folds, default 5).
#   2) A seed loop restricted to multi-study/presplit tasks only (--mode-filter presplit),
#      re-finetuning just those tasks against SEEDS different training seeds so they get a
#      real mean +/- std error bar too, instead of the single-run point estimate from pass 1
#      -- there's no inner hyperparameter search to borrow a std from here (unlike the
#      classical-ML embedding-baselines pipeline), so repeated seeds are the only option.
#      Single-study tasks are skipped in this pass (they already have a real error bar from
#      pass 1's pooled CV, so rerunning them per seed would just waste compute).
#
# Usage:
#   scripts/run_finetune_pretrained_with_seeds.sh [OUTPUT_ROOT]
#
# Env overrides:
#   PRETRAIN_CONFIGS   default: the two winning checkpoints' pretrain configs
#   SEEDS              default: 1 2 3 4 5 -- training seeds for pass 2's presplit-only reruns

set -euo pipefail

OUTPUT_ROOT="${1:-outputs/pretrain/finetune_pretrained_seeds}"

PRETRAIN_CONFIGS="${PRETRAIN_CONFIGS:-configs/pretrain/pretrained_baseline/pretrained_baseline_bins20_mask030.yaml configs/pretrain/real_runs/stage3_abundance_emb_concatenation.yaml}"
SEEDS="${SEEDS:-1 2 3 4 5}"

echo "Re-finetuning ${PRETRAIN_CONFIGS} (pooled CV for single-study tasks, seeded presplit reruns for the rest)"
echo "  Split (fixed): /scratch/kchen13/hmc_final_fixed/downstream_{train,test}.h5ad"
echo "  Results -> ${OUTPUT_ROOT}/<config_name>"

for pretrain_cfg in ${PRETRAIN_CONFIGS}; do
    cfg_name="$(basename "${pretrain_cfg}" .yaml)"
    run_output_root="${OUTPUT_ROOT}/${cfg_name}"
    seeds_root="${run_output_root}_multi_study_seeds"

    echo ""
    echo "=== checkpoint=${cfg_name}: re-finetuning on all tasks (pass 1/2: single run) ==="
    python -m scripts.finetune_all_tasks \
        --config "${pretrain_cfg}" \
        --finetune-output-root "${run_output_root}"

    echo ""
    echo "=== checkpoint=${cfg_name}: re-finetuning presplit tasks across seeds (pass 2/2) ==="
    for seed in ${SEEDS}; do
        echo "  --- seed=${seed} ---"
        python -m scripts.finetune_all_tasks \
            --config "${pretrain_cfg}" \
            --finetune-output-root "${seeds_root}/seed_${seed}" \
            --mode-filter presplit \
            finetune.training.seed="${seed}"
    done

    echo ""
    echo "=== checkpoint=${cfg_name}: aggregating presplit seed results ==="
    python -m scripts.aggregate_seeded_finetune_summary --seeds-root "${seeds_root}"
done

echo ""
echo "Done. See ${OUTPUT_ROOT}/<config_name>/finetune_summary.md (per-task means; cv_summary.yaml"
echo "under each pooled task has the full per-fold spread) and"
echo "${OUTPUT_ROOT}/<config_name>_multi_study_seeds/combined_summary_with_error_bars.md (mean +/-"
echo "std across seeds for location/sex)."
