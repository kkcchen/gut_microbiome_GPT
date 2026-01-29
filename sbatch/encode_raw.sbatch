#!/bin/bash
#SBATCH -A aip-rahulgk
#SBATCH --job-name=encode_raw
#SBATCH --output=logs/encode_raw/%j.out
#SBATCH --error=logs/encode_raw/%j.err
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

# Activate your virtual environment
source ~/hmbenv/bin/activate

python -m scripts.encode_raw \
  --train-input experiment_saves/contrastive_embed/encoded_contrastive_final_depth_aux_train_best.h5ad \
  --test-input experiment_saves/contrastive_embed/encoded_contrastive_final_depth_aux_test_best.h5ad \
  --train-output experiment_saves/contrastive_embed/encoded_contrastive_final_depth_aux_train_best.h5ad \
  --test-output experiment_saves/contrastive_embed/encoded_contrastive_final_depth_aux_test_best.h5ad \
  --prevalence-threshold 0.01 \
  --abundance-threshold 0.05