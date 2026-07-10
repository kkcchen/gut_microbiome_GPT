#!/bin/bash
#SBATCH --job-name=pretrain_stage3
#SBATCH -A aip-brudno
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --array=0-4
#SBATCH --gres=gpu:1
#SBATCH --output=logs/refactored/pretrain/stage3/%a_%j.out
#SBATCH --error=logs/refactored/pretrain/stage3/%a_%j.err

set -euo pipefail

# prepare your environment here
source ~/hmbenv/bin/activate
cd /project/aip-brudno/kchen13/gut_microbiome_GPT

CONFIG_LIST="configs/pretrain/real_runs/stage3_config_list.txt"
CONFIG=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$CONFIG_LIST")

echo "SLURM_JOB_ID=${SLURM_JOB_ID} SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "Using config: ${CONFIG}"

accelerate launch --main_process_port 0 -m scripts.train --config "${CONFIG}"
