#!/bin/bash
#SBATCH -A aip-rahulgk
#SBATCH --job-name=preprocess_hmc
#SBATCH --output=logs/preprocess_hmc/%j.out
#SBATCH --error=logs/preprocess_hmc/%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G

# Activate your virtual environment
source ~/hmbenv/bin/activate

# Arguments
TAXONOMIC_TABLE_PATH="/project/aip-rahulgk/gutmodel/taxonomic_table.csv"
METADATA_PATH="/project/aip-rahulgk/gutmodel/sample_metadata.tsv"

SAVE_DIR="/project/aip-rahulgk/gutmodel/datasets_halfsplit"
NPY_PRETRAIN_FILENAME="taxonomy_table_pretrain.h5ad"
NPY_FINETUNE_FILENAME="taxonomy_table_finetune.h5ad"

# Run the script
echo starting now
python -m scripts.preprocess_hmc \
    --taxonomic_table_path "$TAXONOMIC_TABLE_PATH" \
    --save_dir "$SAVE_DIR" \
    --npy_pretrain_filename "$NPY_PRETRAIN_FILE" \
    --npy_finetune_filename "$NPY_FINETUNE_FILE" \
    --sample_metadata_path "$METADATA_PATH" \
    --seed 42 \
    --split_ratio 0.5 \
    --split_finetune_by_study \
    --add_loc_labels \

echo finished