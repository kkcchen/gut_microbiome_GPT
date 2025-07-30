#!/bin/bash
# this is for local system
# Paths and filenames
TAXONOMIC_TABLE_PATH="/home/kchen/microbiome/gut_microbiome_GPT/datasets/taxonomic_table.csv"
SAVE_DIR="/home/kchen/microbiome/gut_microbiome_GPT/datasets/anndatas"
NPY_PRETRAIN_FILE="taxonomy_table_pretrain.h5ad"
NPY_FINETUNE_FILE="taxonomy_table_finetune.h5ad"
METADATA_PATH="/home/kchen/microbiome/gut_microbiome_GPT/datasets/sample_metadata.tsv"

# Run the script
echo "Starting HMC taxonomic table preprocessing..."

python -m scripts.preprocess_hmc \
    --taxonomic_table_path "$TAXONOMIC_TABLE_PATH" \
    --save_dir "$SAVE_DIR" \
    --npy_pretrain_filename "$NPY_PRETRAIN_FILE" \
    --npy_finetune_filename "$NPY_FINETUNE_FILE" \
    --sample_metadata_path "$METADATA_PATH" \
    --split_finetune_by_study \
    --add_loc_labels \
    --nrows 15000 \

echo "Preprocessing complete."
