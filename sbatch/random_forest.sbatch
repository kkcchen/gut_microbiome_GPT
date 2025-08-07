#!/bin/bash
#SBATCH --job-name=random_forest_microbiome
#SBATCH -A aip-rahulgk
#SBATCH --output=logs/random_forest/%j.out
#SBATCH --error=logs/random_forest/%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00              

# Activate your virtual environment
cd /project/aip-rahulgk/kchen13/gutmodel/gut_microbiome_GPT
source ~/hmbenv/bin/activate

# === Paths to your data ===
# EXPERIMENT_PATH=experiment_saves/taxa-mask-real-mvc
# EMB_SAVE_PATH=$EXPERIMENT_PATH/encoded_samples

# TRAIN_EMBED_PATH=$EMB_SAVE_PATH/encoded_samples_train.h5ad
# TEST_EMBED_PATH=$EMB_SAVE_PATH/encoded_samples_test.h5ad

# below is for encoding raw abundances only.
EXPERIMENT_PATH=experiment_saves/raw_abundances_halfsplit
EMB_SAVE_PATH=/project/aip-rahulgk/gutmodel/datasets_halfsplit/study_split

TRAIN_EMBED_PATH=$EMB_SAVE_PATH/finetune_data_train.h5ad
TEST_EMBED_PATH=$EMB_SAVE_PATH/finetune_data_test.h5ad

OUTPUT_DIR=$EXPERIMENT_PATH/ova_small

NOTES="finetune_data train and test using ova. random search"

# Create a notes.txt file in the output directory with run notes
mkdir -p $OUTPUT_DIR
echo "$NOTES" > $OUTPUT_DIR/notes.txt

# === Run the Random Forest classifier ===
echo "Running Random Forest classification..."
python -m scripts.random_forest_exp \
  --train-embed-path $TRAIN_EMBED_PATH \
  --test-embed-path $TEST_EMBED_PATH \
  --output-dir $OUTPUT_DIR \
  --search-type random \
  --target-colname "location" \
  --emb-name "raw_embedding" \
  # --multiclass \

echo "Random Forest job finished."
