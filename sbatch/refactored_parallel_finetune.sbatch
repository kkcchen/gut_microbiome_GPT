#!/bin/bash
#SBATCH --job-name=finetune_parallel_qwen
#SBATCH -A aip-rahulgk
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=1:00:00
#SBATCH --array=0-111%10
#SBATCH --gres=gpu:1
#SBATCH --output=logs/refactored/finetune/%a_%j.out
#SBATCH --error=logs/refactored/finetune/%a_%j.err

set -euo pipefail

# prepare your environment here
source ~/hmbenv/bin/activate
cd /project/aip-rahulgk/dpellow/gut_microbiome_GPT


CONFIG_LIST="/project/aip-rahulgk/dpellow/gut_microbiome_GPT/configs/finetune/qwen/task_configs/config_list.txt"
CONFIG=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$CONFIG_LIST")

echo "SLURM_JOB_ID=${SLURM_JOB_ID} SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "Using config: ${CONFIG}"

accelerate launch --main_process_port 0 -m scripts.finetune --config "${CONFIG}"
