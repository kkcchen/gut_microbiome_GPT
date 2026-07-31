#!/bin/bash
#SBATCH --job-name=finetune_scratch_with_seeds
#SBATCH -A aip-brudno
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --output=logs/with_seeds/finetune_scratch/%a_%j.out
#SBATCH --error=logs/with_seeds/finetune_scratch/%a_%j.err

# One SLURM array task per random-init baseline config, running
# scripts/run_finetune_scratch_with_seeds.sh restricted to that single config (via its
# SCRATCH_CONFIGS env override) instead of the script's own internal loop over both configs
# sequentially -- lets the 2 configs run as 2 parallel jobs. Each task runs pass 1 (single
# unseeded random-init + finetune, pooled CV for single-study tasks) then pass 2 (5
# random-init + finetune reruns restricted to presplit tasks location/sex via
# --mode-filter), so expect this to be as heavy as the pretrained-checkpoint job.
#
# Usage: sbatch sbatch/run_finetune_scratch_with_seeds.sbatch

set -euo pipefail

source ~/hmbenv/bin/activate
cd /project/aip-brudno/kchen13/gut_microbiome_GPT

SCRATCH_CONFIGS_LIST=(
    "configs/pretrain/real_runs/baseline_random_init_default_scgpt.yaml"
    "configs/pretrain/real_runs/baseline_random_init_winner_arch.yaml"
)
CONFIG="${SCRATCH_CONFIGS_LIST[$SLURM_ARRAY_TASK_ID]}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID} SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "Using config: ${CONFIG}"

SCRATCH_CONFIGS="${CONFIG}" scripts/run_finetune_scratch_with_seeds.sh "outputs/pretrain/real_runs/baseline_seeds"
