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
TAXONOMIC_TABLE_PATH="/projects/gutmodel/taxonomic_table.csv"
PRETRAIN_SAVE_DIR="/projects/gutmodel/datasets/pretrain"
FINETUNE_SAVE_DIR="/projects/gutmodel/datasets/finetune"
NPY_PRETRAIN_FILE="taxonomy_table_pretrain"
NPY_FINETUNE_FILE="taxonomy_table_finetune"
METADATA_PATH="/projects/gutmodel/sample_metadata.tsv"

# Run the script
echo starting now
python -m scripts.preprocess_hmc \
    --taxonomic_table_path "$TAXONOMIC_TABLE_PATH" \
    --pretrain_save_dir "$PRETRAIN_SAVE_DIR" \
    --finetune_save_dir "$FINETUNE_SAVE_DIR" \
    --npy_pretrain_file "$NPY_PRETRAIN_FILE" \
    --npy_finetune_file "$NPY_FINETUNE_FILE"\
    --sample_metadata_path "$METADATA_PATH"\

echo finished