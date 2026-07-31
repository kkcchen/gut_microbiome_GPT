#!/bin/bash
#SBATCH --job-name=finetune_pretrained_with_seeds
#SBATCH -A aip-brudno
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --output=logs/with_seeds/finetune_pretrained/%a_%j.out
#SBATCH --error=logs/with_seeds/finetune_pretrained/%a_%j.err

# One SLURM array task per winning pretrain checkpoint, running
# scripts/run_finetune_pretrained_with_seeds.sh restricted to that single config (via its
# PRETRAIN_CONFIGS env override) instead of the script's own internal loop over both
# checkpoints sequentially -- lets the 2 checkpoints run as 2 parallel jobs. Each task runs
# pass 1 (single unseeded run, pooled CV for single-study tasks) then pass 2 (5 training-seed
# reruns restricted to presplit tasks location/sex via --mode-filter), so expect this to be
# the heaviest of the three with_seeds jobs.
#
# Usage: sbatch sbatch/run_finetune_pretrained_with_seeds.sbatch

set -euo pipefail

source ~/hmbenv/bin/activate
cd /project/aip-brudno/kchen13/gut_microbiome_GPT

PRETRAIN_CONFIGS_LIST=(
    "configs/pretrain/pretrained_baseline/pretrained_baseline_bins20_mask030.yaml"
    "configs/pretrain/real_runs/stage3_abundance_emb_concatenation.yaml"
)
CONFIG="${PRETRAIN_CONFIGS_LIST[$SLURM_ARRAY_TASK_ID]}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID} SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "Using config: ${CONFIG}"

PRETRAIN_CONFIGS="${CONFIG}" scripts/run_finetune_pretrained_with_seeds.sh "outputs/pretrain/finetune_pretrained_seeds"
