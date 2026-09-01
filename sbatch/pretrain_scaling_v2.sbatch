#!/bin/bash
#SBATCH --job-name=pretrain_scaling_v2
#SBATCH -A aip-brudno
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --array=0-7
#SBATCH --gres=gpu:1
#SBATCH --output=logs/refactored/pretrain/scaling_v2/%a_%j.out
#SBATCH --error=logs/refactored/pretrain/scaling_v2/%a_%j.err

# Data-scaling v2 pretraining: 8 runs (2 recipes x 4 ratios) on the study-isolated
# splits from scripts/make_study_scaling_manifests.py. Fresh output tree
# (outputs/pretrain/scaling_v2/) so scripts/train.py's skip-guard never fires and the
# v1 results are untouched.
#
# --tasks __none__ makes the inline auto-finetune skip every task (recorded as
# "skipped", so the job still exits 0): downstream eval for v2 is full-finetune only,
# run separately by sbatch/finetune_scaling_v2.sbatch.

set -euo pipefail

source ~/hmbenv/bin/activate
cd /project/aip-brudno/kchen13/gut_microbiome_GPT
mkdir -p logs/refactored/pretrain/scaling_v2

CONFIG_LIST="configs/pretrain/scaling/scaling_v2_config_list.txt"
CONFIG=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$CONFIG_LIST")

echo "SLURM_JOB_ID=${SLURM_JOB_ID} SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "Using config: ${CONFIG}"

accelerate launch --main_process_port 0 -m scripts.train --config "${CONFIG}" --tasks __none__
