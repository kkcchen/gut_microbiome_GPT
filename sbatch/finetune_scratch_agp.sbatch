#!/bin/bash
#SBATCH --job-name=finetune_scratch_agp
#SBATCH -A aip-brudno
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --output=logs/with_seeds/finetune_scratch_agp/%a_%j.out
#SBATCH --error=logs/with_seeds/finetune_scratch_agp/%a_%j.err

# One SLURM array task per random-init baseline config, running scripts/train.py (random-init
# encoder, max_epochs=0, immediately followed by the auto-finetune chain) on the 18 AGP tasks
# in configs/finetune/task_registry.yaml, scoped with --tasks so the other (HMC) tasks in
# that same registry are skipped rather than pointed at the wrong data -- see
# utils/finetune_orchestration.py::run_all_downstream_finetunes' tasks param.
#
# Single pooled-CV pass only, no seed loop: every AGP task is single-study (see
# scripts/split_agp_downstream_tasks.py), so utils/downstream_split_mode.py::resolve_split_mode
# resolves all of them to combined_cv -- a real 5-fold error bar comes for free, same as
# sbatch/finetune_pretrained_agp.sbatch / sbatch/run_embedding_baselines_agp.sbatch.
#
# Usage: sbatch sbatch/finetune_scratch_agp.sbatch

set -euo pipefail

source ~/hmbenv/bin/activate
cd /project/aip-brudno/kchen13/gut_microbiome_GPT

SCRATCH_CONFIGS_LIST=(
    "configs/pretrain/real_runs/baseline_random_init_default_scgpt.yaml"
    "configs/pretrain/real_runs/baseline_random_init_winner_arch.yaml"
)
CONFIG="${SCRATCH_CONFIGS_LIST[$SLURM_ARRAY_TASK_ID]}"
CFG_NAME="$(basename "${CONFIG}" .yaml)"

# --tasks takes one comma-separated string (not repeated flag values): scripts/train.py also
# has a REMAINDER positional for the config overrides below, and an nargs='*' --tasks would
# greedily swallow those too since they don't start with '-'.
AGP_TASKS="age cat,alcohol_frequency,antibiotic_history,asd,autoimmune,contraceptive,diabetes,fruit_frequency,gluten,ibd,ibs,meat_eggs_frequency,probiotic_frequency,sibo,sleep_duration,sugary_sweets_frequency,vegetable_frequency,weight_change"

echo "SLURM_JOB_ID=${SLURM_JOB_ID} SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "Using config: ${CONFIG}"

python -m scripts.train \
    --config "${CONFIG}" \
    --tasks "${AGP_TASKS}" \
    paths.output_dir="outputs/pretrain/real_runs/baseline_agp/${CFG_NAME}" \
    finetune.paths.downstream_train=/scratch/kchen13/hmc_final_fixed/downstream_agp_train.h5ad \
    finetune.paths.downstream_test=/scratch/kchen13/hmc_final_fixed/downstream_agp_test.h5ad
