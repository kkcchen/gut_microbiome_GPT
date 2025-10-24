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
  --train-input /project/aip-rahulgk/gutmodel/remove_redundant_oct24/finetune_train.h5ad \
  --test-input /project/aip-rahulgk/gutmodel/remove_redundant_oct24/finetune_test.h5ad \
  --prevalence-threshold 0.01 \
  --abundance-threshold 0.05