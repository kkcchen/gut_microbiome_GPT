#!/bin/bash
#SBATCH --job-name=finetune_pretrained_agp_full_resume
#SBATCH -A aip-brudno
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/finetune_pretrained_agp_full/resume_%A_%a.out
#SBATCH --error=logs/finetune_pretrained_agp_full/resume_%A_%a.err
#SBATCH --array=0-1

# Assembly step for finetune_pretrained_agp_full.sbatch's 180 fold jobs. No --fold:
# each of these 2 jobs (one per checkpoint) walks all 18 tasks' 5 folds, cache-hits
# every one of them, and writes each task's cv_summary.yaml -- no retraining, fast.
#
# Usage: sbatch sbatch/finetune_pretrained_agp_full_resume.sbatch

set -uo pipefail

source ~/hmbenv/bin/activate
cd /project/aip-brudno/kchen13/gut_microbiome_GPT

PRETRAIN_CONFIGS=(
    "configs/pretrain/pretrained_baseline/pretrained_baseline_bins20_mask030.yaml"
    "configs/pretrain/real_runs/stage3_abundance_emb_concatenation.yaml"
)
CONFIG="${PRETRAIN_CONFIGS[$SLURM_ARRAY_TASK_ID]}"
CFG_NAME="$(basename "${CONFIG}" .yaml)"

AGP_TASKS="age cat,alcohol_frequency,antibiotic_history,asd,autoimmune,contraceptive,diabetes,fruit_frequency,gluten,ibd,ibs,meat_eggs_frequency,probiotic_frequency,sibo,sleep_duration,sugary_sweets_frequency,vegetable_frequency,weight_change"

echo "array index $SLURM_ARRAY_TASK_ID -> config '$CFG_NAME'"

python -m scripts.finetune_all_tasks \
    --config "${CONFIG}" \
    --finetune-output-root "outputs/pretrain/finetune_pretrained_agp_full/${CFG_NAME}" \
    --tasks "${AGP_TASKS}" \
    finetune.paths.downstream_train=/scratch/kchen13/hmc_final_fixed/downstream_agp_train.h5ad \
    finetune.paths.downstream_test=/scratch/kchen13/hmc_final_fixed/downstream_agp_test.h5ad \
    finetune.training.finetune_mode=full
