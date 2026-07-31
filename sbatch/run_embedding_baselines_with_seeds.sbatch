#!/bin/bash
#SBATCH --job-name=embedding_baselines_with_seeds
#SBATCH -A aip-brudno
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --output=logs/with_seeds/embedding_baselines/%a_%j.out
#SBATCH --error=logs/with_seeds/embedding_baselines/%a_%j.err

# One SLURM array task per config in configs/eval/embedding_baselines/config_list.txt,
# running scripts/run_embedding_baselines_with_seeds.sh restricted to that single config
# (via its EMBED_CONFIGS env override) instead of the script's own internal loop over both
# configs sequentially -- lets the 2 configs run as 2 parallel jobs.
#
# Usage: sbatch sbatch/run_embedding_baselines_with_seeds.sbatch

set -euo pipefail

source ~/hmbenv/bin/activate
cd /project/aip-brudno/kchen13/gut_microbiome_GPT

CONFIG_LIST="configs/eval/embedding_baselines/config_list.txt"
CONFIG=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$CONFIG_LIST")

echo "SLURM_JOB_ID=${SLURM_JOB_ID} SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "Using config: ${CONFIG}"

EMBED_CONFIGS="${CONFIG}" scripts/run_embedding_baselines_with_seeds.sh "outputs/eval/embedding_baselines_seeds"
