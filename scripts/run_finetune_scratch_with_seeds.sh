#!/usr/bin/env bash
# Runs the "finetuned from scratch" (no-pretraining) baselines -- max_epochs=0 random-init
# encoder, immediately followed by the auto-finetune-all-tasks chain -- against the fixed
# split (/scratch/kchen13/hmc_final_fixed/downstream_{train,test}.h5ad).
#
# Two passes per config:
#   1) One full run, no seed loop: single-study tasks (no natural train/test boundary) are
#      pooled and K-fold cross-validated by utils/finetune_cv_utils.py::run_task_cv (one
#      training run per fold, each with its own random-init encoder), so the error bar comes
#      from the CV folds themselves -- one run, one seed is enough. Multi-study tasks
#      (location, sex -- disjoint train/test studies, a real cross-study holdout) also run
#      once here, giving the usual single point estimate. See utils/downstream_split_mode.py
#      and configs/finetune/task_registry.yaml for the mode detection and CV settings
#      (outer_folds, default 5) -- expect roughly 4x the training runs per config compared
#      to the old always-presplit behavior.
#   2) A seed loop restricted to multi-study/presplit tasks only (--mode-filter presplit),
#      rerunning the (cheap, max_epochs=0) random-init + finetune chain SEEDS times so those
#      tasks get a real mean +/- std error bar too, instead of the single-run point estimate
#      from pass 1 -- there's no inner hyperparameter search to borrow a std from here
#      (unlike the classical-ML embedding-baselines pipeline). Each seed writes to its own
#      paths.output_dir, so it also gets its own fresh random-init encoder, not just a
#      different finetune seed. Single-study tasks are skipped in this pass (they already
#      have a real error bar from pass 1's pooled CV, so rerunning them per seed would just
#      waste compute).
#
# Unlike scripts/run_finetune_pretrained_with_seeds.sh, this reruns scripts/train.py itself
# (unmodified aside from --mode-filter) -- there's no expensive pretraining to avoid
# repeating here (max_epochs=0), so no separate re-finetune-only script is needed.
#
# Usage:
#   scripts/run_finetune_scratch_with_seeds.sh [OUTPUT_ROOT]
#
# Env overrides:
#   SCRATCH_CONFIGS   default: the two baseline_random_init_*.yaml configs
#   SEEDS             default: 1 2 3 4 5 -- training seeds for pass 2's presplit-only reruns

set -euo pipefail

OUTPUT_ROOT="${1:-outputs/pretrain/real_runs/baseline_seeds}"

SCRATCH_CONFIGS="${SCRATCH_CONFIGS:-configs/pretrain/real_runs/baseline_random_init_default_scgpt.yaml configs/pretrain/real_runs/baseline_random_init_winner_arch.yaml}"
SEEDS="${SEEDS:-1 2 3 4 5}"

echo "Running from-scratch finetune baselines (pooled CV for single-study tasks, seeded presplit reruns for the rest)"
echo "  Split (fixed): /scratch/kchen13/hmc_final_fixed/downstream_{train,test}.h5ad"
echo "  Results -> ${OUTPUT_ROOT}/<config_name>"

for scratch_cfg in ${SCRATCH_CONFIGS}; do
    cfg_name="$(basename "${scratch_cfg}" .yaml)"
    run_output_dir="${OUTPUT_ROOT}/${cfg_name}"
    seeds_root="${run_output_dir}_multi_study_seeds"

    echo ""
    echo "=== config=${cfg_name}: random-init + finetune on all tasks (pass 1/2: single run) ==="
    python -m scripts.train \
        --config "${scratch_cfg}" \
        paths.output_dir="${run_output_dir}"

    echo ""
    echo "=== config=${cfg_name}: random-init + finetune presplit tasks across seeds (pass 2/2) ==="
    for seed in ${SEEDS}; do
        echo "  --- seed=${seed} ---"
        python -m scripts.train \
            --config "${scratch_cfg}" \
            --mode-filter presplit \
            paths.output_dir="${seeds_root}/seed_${seed}" \
            training.seed="${seed}" \
            finetune.training.seed="${seed}"
    done

    echo ""
    echo "=== config=${cfg_name}: aggregating presplit seed results ==="
    python -m scripts.aggregate_seeded_finetune_summary --seeds-root "${seeds_root}"
done

echo ""
echo "Done. See ${OUTPUT_ROOT}/<config_name>/finetune_summary.md (per-task means; cv_summary.yaml"
echo "under each pooled task has the full per-fold spread) and"
echo "${OUTPUT_ROOT}/<config_name>_multi_study_seeds/combined_summary_with_error_bars.md (mean +/-"
echo "std across seeds for location/sex)."
