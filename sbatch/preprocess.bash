#!/bin/bash
# this is for local system
# Paths and filenames
TAXONOMIC_TABLE_PATH="/home/kchen/microbiome/gut_microbiome_GPT/datasets/taxonomic_table.csv"
PRETRAIN_SAVE_DIR="/home/kchen/microbiome/gut_microbiome_GPT/datasets/pretrain"
FINETUNE_SAVE_DIR="/home/kchen/microbiome/gut_microbiome_GPT/datasets/finetune"
NPY_PRETRAIN_FILE="taxonomy_table_pretrain.npy"
NPY_FINETUNE_FILE="taxonomy_table_finetune.npy"
METADATA_PATH="/home/kchen/microbiome/gut_microbiome_GPT/datasets/sample_metadata.tsv"

# Run the script
echo "Starting HMC taxonomic table preprocessing..."

cd /project/aip-rahulgk/kchen13/gutmodel/gut_microbiome_GPT
source ~/hmbenv/bin/activate

python -m scripts.preprocess_hmc \
    --taxonomic_table_path "$TAXONOMIC_TABLE_PATH" \
    --pretrain_save_dir "$PRETRAIN_SAVE_DIR" \
    --finetune_save_dir "$FINETUNE_SAVE_DIR" \
    --npy_pretrain_filename "$NPY_PRETRAIN_FILE" \
    --npy_finetune_filename "$NPY_FINETUNE_FILE" \
    --sample_metadata_path "$METADATA_PATH"

echo "Preprocessing complete."
