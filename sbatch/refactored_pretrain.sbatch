#!/bin/bash
#SBATCH --job-name=pretrain_masking_experiments
#SBATCH -A aip-rahulgk
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=10:00:00
#SBATCH --output=logs/refactored/pretrain/%j.out
#SBATCH --error=logs/refactored/pretrain/%j.err
#SBATCH --gres=gpu:1

# prepare your environment here
source ~/hmbenv/bin/activate
cd /project/aip-rahulgk/dpellow/gut_microbiome_GPT


python -m wandb login 7559085cf6a63fa75982fe12e737bdd576c4fc62

accelerate launch --main_process_port 0 -m scripts.train \
    --config configs/pretrain/contrastive_config.yaml
