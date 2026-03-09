#!/bin/bash
#SBATCH --job-name=pretrain_parallel_denoising_contrastive
#SBATCH -A aip-rahulgk
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=6:45:00
#SBATCH --array=0-71%10
#SBATCH --gres=gpu:1
#SBATCH --output=logs/refactored/pretrain/%a_%j.out
#SBATCH --error=logs/refactored/pretrain/%a_%j.err

set -euo pipefail

# prepare your environment here
source ~/hmbenv/bin/activate
cd /project/aip-rahulgk/dpellow/gut_microbiome_GPT


CONFIG_LIST="/project/aip-rahulgk/dpellow/gut_microbiome_GPT/configs/pretrain/denoising_contrastive/config_list.txt"
CONFIG=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$CONFIG_LIST")

echo "SLURM_JOB_ID=${SLURM_JOB_ID} SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "Using config: ${CONFIG}"

accelerate launch --main_process_port 0 -m scripts.train --config "${CONFIG}"
