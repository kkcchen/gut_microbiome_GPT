#!/bin/bash
#SBATCH --job-name=preprocess_hmc
#SBATCH --output=logs/preprocess_hmc_%j.out
#SBATCH --error=logs/preprocess_hmc_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Load any required modules
module load python/3.10.12

# Activate your virtual environment
source ~/hmbenv/bin/activate

# Arguments
TAXONOMIC_TABLE_PATH="~/datasets/taxonomic_table.csv"
PRETRAIN_SAVE_DIR="~/gut_microbiome_GPT/datasets/pretrain"
FINETUNE_SAVE_DIR="~/gut_microbiome_GPT/datasets/finetune"
NPY_PRETRAIN_FILE="taxonomic_table_pretrain.npy"
NPY_FINETUNE_FILE="taxonomic_table_finetune.npy"

# Run the script
python scripts/preprocess_hmc.py \
    --taxonomic_table_path "$TAXONOMIC_TABLE_PATH" \
    --pretrain_save_dir "$PRETRAIN_SAVE_DIR" \
    --finetune_save_dir "$FINETUNE_SAVE_DIR" \
    --npy_pretrain_file "$NPY_PRETRAIN_FILE" \
    --npy_finetune_file "$NPY_FINETUNE_FILE"
