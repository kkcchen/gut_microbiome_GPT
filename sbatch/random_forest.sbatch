#!/bin/bash
#SBATCH --job-name=random_forest_microbiome
#SBATCH --output=logs/random_forest/%j.out
#SBATCH --error=logs/random_forest/%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3:00:00              
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Activate your virtual environment
cd ~/gut_microbiome_GPT
source ~/hmbenv/bin/activate

# === Paths to your data ===
TRAIN_EMBED_PATH=encoded_samples/encoded_samples_train.npy
TRAIN_LABELS_PATH=datasets/finetune/train/finetune_locs_train.json

TEST_EMBED_PATH=encoded_samples/encoded_samples_test.npy
TEST_LABELS_PATH=datasets/finetune/test/finetune_locs_test.json

# === Run the Random Forest classifier ===
echo "Running Random Forest classification..."
python scripts/random_forest_transformer.py \
  --train-embed-path $TRAIN_EMBED_PATH \
  --train-loc-labels-path $TRAIN_LABELS_PATH \
  --test-embed-path $TEST_EMBED_PATH \
  --test-loc-labels-path $TEST_LABELS_PATH

echo "Random Forest job finished."
