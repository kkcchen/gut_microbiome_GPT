#!/bin/bash
#SBATCH --job-name=finetune_pretrained_orig_full_resume
#SBATCH -A aip-brudno
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/finetune_pretrained_orig_full/resume_%A_%a.out
#SBATCH --error=logs/finetune_pretrained_orig_full/resume_%A_%a.err
#SBATCH --array=0-1

# Assembly step for finetune_pretrained_orig_full.sbatch's 90 fold jobs. No --fold,
# scoped to just the 9 combined_cv tasks (not location/sex -- see
# finetune_pretrained_orig_full_presplit.sbatch for those, run separately so this
# assembly pass never retrains them). Cache-hits every fold, writes cv_summary.yaml.
#
# Usage: sbatch sbatch/finetune_pretrained_orig_full_resume.sbatch

set -uo pipefail

source ~/hmbenv/bin/activate
cd /project/aip-brudno/kchen13/gut_microbiome_GPT

PRETRAIN_CONFIGS=(
    "configs/pretrain/pretrained_baseline/pretrained_baseline_bins20_mask030.yaml"
    "configs/pretrain/real_runs/stage3_abundance_emb_concatenation.yaml"
)
CONFIG="${PRETRAIN_CONFIGS[$SLURM_ARRAY_TASK_ID]}"
CFG_NAME="$(basename "${CONFIG}" .yaml)"

ORIG_COMBINED_CV_TASKS="chemotherapy,crohns_disease,diarrhea,hbv_infection,helicobacter_pylori_infection,hiv_infection,sunflower_seed_oil_emollient_therapy,supplement,type_1_diabetes"

echo "array index $SLURM_ARRAY_TASK_ID -> config '$CFG_NAME'"

python -m scripts.finetune_all_tasks \
    --config "${CONFIG}" \
    --finetune-output-root "outputs/pretrain/finetune_pretrained_orig_full/${CFG_NAME}" \
    --tasks "${ORIG_COMBINED_CV_TASKS}" \
    finetune.paths.downstream_train=/scratch/kchen13/hmc_final_fixed/downstream_train.h5ad \
    finetune.paths.downstream_test=/scratch/kchen13/hmc_final_fixed/downstream_test.h5ad \
    finetune.training.finetune_mode=full
