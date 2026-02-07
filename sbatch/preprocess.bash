#!/bin/bash
#SBATCH --job-name=microbiome_preprocess
#SBATCH -A aip-rahulgk
#SBATCH --output=logs/preprocess/%j.out
#SBATCH --error=logs/preprocess/%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00   

# Paths and filenames
TAXONOMIC_TABLE_PATH="/home/tiantian/projects/aip-rahulgk/tiantian/gut_microbiome_GPT/data/taxonomic_table.csv"
SAVE_DIR="/home/tiantian/projects/aip-rahulgk/tiantian/gut_microbiome_GPT/data/hmc_filtered_with_downstream"
ANN_PRETRAIN_FILE="taxonomy_table_pretrain.h5ad"
ANN_FINETUNE_FILE="taxonomy_table_finetune.h5ad"
METADATA_PATH="/home/tiantian/projects/aip-rahulgk/tiantian/gut_microbiome_GPT/data/sample_metadata.tsv"
TAGS_PATH="/home/tiantian/projects/aip-rahulgk/tiantian/gut_microbiome_GPT/data/tags.tsv"

# Run the script
echo "Starting HMC taxonomic table preprocessing..."

source ~/hmbenv/bin/activate
cd /project/aip-rahulgk/tiantian/gut_microbiome_GPT

python -m scripts.preprocess_hmc \
    --taxonomic_table_path "$TAXONOMIC_TABLE_PATH" \
    --save_dir "$SAVE_DIR" \
    --sample_metadata_path "$METADATA_PATH" \
    --tags_path "$TAGS_PATH" \
    --split_finetune_by_study \
    --add_loc_labels \

echo "Preprocessing complete."
