# #!/bin/bash
# # this is for local system
# # Paths and filenames
# TAXONOMIC_TABLE_PATH="/home/kchen/microbiome/gut_microbiome_GPT/datasets/taxonomic_table.csv"
# SAVE_DIR="/home/kchen/microbiome/gut_microbiome_GPT/datasets/anndatas"
# ANN_PRETRAIN_FILE="taxonomy_table_pretrain.h5ad"
# ANN_FINETUNE_FILE="taxonomy_table_finetune.h5ad"
# METADATA_PATH="/home/kchen/microbiome/gut_microbiome_GPT/datasets/sample_metadata.tsv"

# # Run the script
# echo "Starting HMC taxonomic table preprocessing..."

# cd /project/aip-rahulgk/kchen13/gutmodel/gut_microbiome_GPT
# source ~/hmbenv/bin/activate

# python -m scripts.preprocess_hmc \
#     --taxonomic_table_path "$TAXONOMIC_TABLE_PATH" \
#     --save_dir "$SAVE_DIR" \
#     --anndata_pretrain_filename "$ANN_PRETRAIN_FILE" \
#     --anndata_finetune_filename "$ANN_FINETUNE_FILE" \
#     --sample_metadata_path "$METADATA_PATH" \
#     --split_finetune_by_study \
#     --add_loc_labels \
#     --nrows 15000 \

# echo "Preprocessing complete."
