#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --job-name=preprocess_hmc
#SBATCH --output=logs/preprocess_hmc/%j.out
#SBATCH --error=logs/preprocess_hmc/%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --qos=cpu_qos
#SBATCH --partition=cpu

# Activate your virtual environment
source ~/hmbenv/bin/activate

# Arguments
TAXONOMIC_TABLE_PATH="$HOME/datasets/taxonomic_table.csv"
PRETRAIN_SAVE_DIR="$HOME/gut_microbiome_GPT/datasets/pretrain"
FINETUNE_SAVE_DIR="$HOME/gut_microbiome_GPT/datasets/finetune"
NPY_PRETRAIN_FILE="taxonomy_table_pretrain"
NPY_FINETUNE_FILE="taxonomy_table_finetune"

# Run the script
echo starting now
python scripts/preprocess_hmc.py \
    --taxonomic_table_path "$TAXONOMIC_TABLE_PATH" \
    --pretrain_save_dir "$PRETRAIN_SAVE_DIR" \
    --finetune_save_dir "$FINETUNE_SAVE_DIR" \
    --npy_pretrain_file "$NPY_PRETRAIN_FILE" \
    --npy_finetune_file "$NPY_FINETUNE_FILE"

echo finished