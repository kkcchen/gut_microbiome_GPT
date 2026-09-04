#!/bin/bash
#SBATCH --job-name=finetune_pretrained_agp_full
#SBATCH -A aip-brudno
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/finetune_pretrained_agp_full/folds_%A_%a.out
#SBATCH --error=logs/finetune_pretrained_agp_full/folds_%A_%a.err
#SBATCH --array=0-179

# Fold-parallel finetune_mode=full ablation on the 18 AGP tasks, both winning
# pretrained checkpoints. Same data/tasks/CV as sbatch/finetune_pretrained_agp.sbatch,
# except finetune.training.finetune_mode=full (unfreeze and train the whole
# transformer) instead of that script's default "partial" (linear probe on a frozen
# transformer) -- isolates what letting the pretrained weights move actually buys.
#
# One array job per (checkpoint, task, fold) instead of one job per checkpoint
# working through all 18 tasks x 5 folds in sequence (what finetune_pretrained_agp.sbatch
# does): each fold is a full finetuning run, so this bounds a job to one fold's
# wall-clock instead of 90. Needs the --fold support added to
# utils/finetune_cv_utils.py::run_task_cv / utils/finetune_orchestration.py /
# scripts/finetune_all_tasks.py -- each job trains (or skips, if already cached)
# exactly one outer fold under
# outputs/pretrain/finetune_pretrained_agp_full/<cfg>/finetune/<task>/fold_<i>/
# and does NOT write the task's cv_summary.yaml.
#
# 2 checkpoints x 18 tasks x 5 folds = 180, array 0-179.
#
# After all 180 finish, run finetune_pretrained_agp_full_resume.sbatch to assemble
# (cache-hits every fold, so that's fast).
#
# Usage: sbatch sbatch/finetune_pretrained_agp_full.sbatch

set -uo pipefail

source ~/hmbenv/bin/activate
cd /project/aip-brudno/kchen13/gut_microbiome_GPT

PRETRAIN_CONFIGS=(
    "configs/pretrain/pretrained_baseline/pretrained_baseline_bins20_mask030.yaml"
    "configs/pretrain/real_runs/stage3_abundance_emb_concatenation.yaml"
)

AGP_TASKS=(
    "age cat"
    "alcohol_frequency"
    "antibiotic_history"
    "asd"
    "autoimmune"
    "contraceptive"
    "diabetes"
    "fruit_frequency"
    "gluten"
    "ibd"
    "ibs"
    "meat_eggs_frequency"
    "probiotic_frequency"
    "sibo"
    "sleep_duration"
    "sugary_sweets_frequency"
    "vegetable_frequency"
    "weight_change"
)

N_TASKS=${#AGP_TASKS[@]}
N_FOLDS=5
PER_CONFIG=$(( N_TASKS * N_FOLDS ))

config_idx=$(( SLURM_ARRAY_TASK_ID / PER_CONFIG ))
remainder=$(( SLURM_ARRAY_TASK_ID % PER_CONFIG ))
task_idx=$(( remainder / N_FOLDS ))
fold=$(( remainder % N_FOLDS ))

CONFIG="${PRETRAIN_CONFIGS[$config_idx]}"
CFG_NAME="$(basename "${CONFIG}" .yaml)"
TASK="${AGP_TASKS[$task_idx]}"

echo "array index $SLURM_ARRAY_TASK_ID -> config '$CFG_NAME', task '$TASK', fold $fold"

python -m scripts.finetune_all_tasks \
    --config "${CONFIG}" \
    --finetune-output-root "outputs/pretrain/finetune_pretrained_agp_full/${CFG_NAME}" \
    --tasks "${TASK}" \
    --fold "${fold}" \
    finetune.paths.downstream_train=/scratch/kchen13/hmc_final_fixed/downstream_agp_train.h5ad \
    finetune.paths.downstream_test=/scratch/kchen13/hmc_final_fixed/downstream_agp_test.h5ad \
    finetune.training.finetune_mode=full
