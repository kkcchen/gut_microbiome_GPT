#!/bin/bash
#SBATCH --job-name=eval_contrastive
#SBATCH -A aip-rahulgk
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=10:00:00
#SBATCH --output=logs/refactored/finetune/%j.out
#SBATCH --error=logs/refactored/finetune/%j.err
#SBATCH --gres=gpu:1

# prepare your environment here
source ~/hmbenv/bin/activate
cd /project/aip-rahulgk/dpellow/gut_microbiome_GPT


accelerate launch --main_process_port 0 -m scripts.finetune \
    --config configs/finetune/david_finetune.yaml
