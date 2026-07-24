#!/bin/bash
#SBATCH --job-name=eval_raw_baselines
#SBATCH -A aip-brudno
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32
#SBATCH --time=5:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/refactored/eval/raw_baselines/%a_%j.out
#SBATCH --error=logs/refactored/eval/raw_baselines/%a_%j.err

set -euo pipefail

# prepare your environment here
source ~/hmbenv/bin/activate
cd /project/aip-brudno/kchen13/gut_microbiome_GPT

CONFIG_LIST="configs/eval/raw_baselines/config_list.txt"
CONFIG=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$CONFIG_LIST")

echo "SLURM_JOB_ID=${SLURM_JOB_ID} SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "Using config: ${CONFIG}"

accelerate launch --main_process_port 0 -m scripts.eval_downstream_only --config "${CONFIG}"
