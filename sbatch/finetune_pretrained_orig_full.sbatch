#!/bin/bash
#SBATCH --job-name=finetune_pretrained_orig_full
#SBATCH -A aip-brudno
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/finetune_pretrained_orig_full/folds_%A_%a.out
#SBATCH --error=logs/finetune_pretrained_orig_full/folds_%A_%a.err
#SBATCH --array=0-89

# Fold-parallel finetune_mode=full ablation on the 9 single-study (combined_cv)
# original HMC tasks, both winning pretrained checkpoints -- same idea as
# finetune_pretrained_agp_full.sbatch, see that file for the rationale.
#
# location/sex are deliberately NOT in this array: they're presplit (disjoint
# train/test studies), so they have no CV folds to parallelize -- see
# finetune_pretrained_orig_full_presplit.sbatch for those two.
#
# 2 checkpoints x 9 tasks x 5 folds = 90, array 0-89.
#
# After all 90 finish, run finetune_pretrained_orig_full_resume.sbatch to assemble.
#
# Usage: sbatch sbatch/finetune_pretrained_orig_full.sbatch

set -uo pipefail

source ~/hmbenv/bin/activate
cd /project/aip-brudno/kchen13/gut_microbiome_GPT

PRETRAIN_CONFIGS=(
    "configs/pretrain/pretrained_baseline/pretrained_baseline_bins20_mask030.yaml"
    "configs/pretrain/real_runs/stage3_abundance_emb_concatenation.yaml"
)

ORIG_COMBINED_CV_TASKS=(
    "chemotherapy"
    "crohns_disease"
    "diarrhea"
    "hbv_infection"
    "helicobacter_pylori_infection"
    "hiv_infection"
    "sunflower_seed_oil_emollient_therapy"
    "supplement"
    "type_1_diabetes"
)

N_TASKS=${#ORIG_COMBINED_CV_TASKS[@]}
N_FOLDS=5
PER_CONFIG=$(( N_TASKS * N_FOLDS ))

config_idx=$(( SLURM_ARRAY_TASK_ID / PER_CONFIG ))
remainder=$(( SLURM_ARRAY_TASK_ID % PER_CONFIG ))
task_idx=$(( remainder / N_FOLDS ))
fold=$(( remainder % N_FOLDS ))

CONFIG="${PRETRAIN_CONFIGS[$config_idx]}"
CFG_NAME="$(basename "${CONFIG}" .yaml)"
TASK="${ORIG_COMBINED_CV_TASKS[$task_idx]}"

echo "array index $SLURM_ARRAY_TASK_ID -> config '$CFG_NAME', task '$TASK', fold $fold"

python -m scripts.finetune_all_tasks \
    --config "${CONFIG}" \
    --finetune-output-root "outputs/pretrain/finetune_pretrained_orig_full/${CFG_NAME}" \
    --tasks "${TASK}" \
    --fold "${fold}" \
    finetune.paths.downstream_train=/scratch/kchen13/hmc_final_fixed/downstream_train.h5ad \
    finetune.paths.downstream_test=/scratch/kchen13/hmc_final_fixed/downstream_test.h5ad \
    finetune.training.finetune_mode=full
