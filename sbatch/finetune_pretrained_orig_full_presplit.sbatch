#!/bin/bash
#SBATCH --job-name=finetune_pretrained_orig_full_presplit
#SBATCH -A aip-brudno
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/finetune_pretrained_orig_full/presplit_%A_%a.out
#SBATCH --error=logs/finetune_pretrained_orig_full/presplit_%A_%a.err
#SBATCH --array=0-1

# finetune_mode=full for location/sex -- the 2 presplit (cross-study) original
# tasks, which have no CV folds to parallelize (real train/test study boundary,
# not a resampled split), so each just runs once per checkpoint here. Separate
# from finetune_pretrained_orig_full.sbatch/_resume.sbatch (which only ever
# touch the 9 combined_cv tasks) so this never collides with or re-triggers that
# fold-parallel work.
#
# 2 checkpoints x {location, sex} in one job each = array 0-1.
#
# Usage: sbatch sbatch/finetune_pretrained_orig_full_presplit.sbatch

set -uo pipefail

source ~/hmbenv/bin/activate
cd /project/aip-brudno/kchen13/gut_microbiome_GPT

PRETRAIN_CONFIGS=(
    "configs/pretrain/pretrained_baseline/pretrained_baseline_bins20_mask030.yaml"
    "configs/pretrain/real_runs/stage3_abundance_emb_concatenation.yaml"
)
CONFIG="${PRETRAIN_CONFIGS[$SLURM_ARRAY_TASK_ID]}"
CFG_NAME="$(basename "${CONFIG}" .yaml)"

echo "array index $SLURM_ARRAY_TASK_ID -> config '$CFG_NAME', tasks location,sex"

python -m scripts.finetune_all_tasks \
    --config "${CONFIG}" \
    --finetune-output-root "outputs/pretrain/finetune_pretrained_orig_full/${CFG_NAME}" \
    --tasks "location,sex" \
    --mode-filter presplit \
    finetune.paths.downstream_train=/scratch/kchen13/hmc_final_fixed/downstream_train.h5ad \
    finetune.paths.downstream_test=/scratch/kchen13/hmc_final_fixed/downstream_test.h5ad \
    finetune.training.finetune_mode=full
