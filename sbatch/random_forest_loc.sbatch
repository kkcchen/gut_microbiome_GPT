#!/bin/bash
#SBATCH --job-name=random_forest_testonly
#SBATCH -A aip-rahulgk
#SBATCH --output=logs/random_forest_testonly/%j.out
#SBATCH --error=logs/random_forest_testonly/%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00              

# Activate your virtual environment
cd /project/aip-rahulgk/kchen13/gutmodel/gut_microbiome_GPT
source ~/hmbenv/bin/activate

FROM_RAW=1

if [ "$FROM_RAW" -eq 1 ]; then
  echo "Using raw embeddings..."
  EXPERIMENT_PATH=experiment_saves/raw_abundances_newsplit
  EMB_SAVE_PATH=/project/aip-rahulgk/gutmodel/datasets_newsplit

  TRAIN_EMBED_PATH=$EMB_SAVE_PATH/finetune_loc_train.h5ad
  TEST_EMBED_PATH=$EMB_SAVE_PATH/finetune_loc_test.h5ad

  OUTPUT_DIR="$EXPERIMENT_PATH/loc_raw_quick"

  EMB_NAME=raw_embedding
else
  echo "Using pre-encoded embeddings..."
  EXPERIMENT_PATH=experiment_saves/taxa-mask-30-locexc
  EMB_SAVE_PATH=$EXPERIMENT_PATH/encoded_samples

  TRAIN_EMBED_PATH=$EMB_SAVE_PATH/encoded_loc_train_finetune.h5ad
  TEST_EMBED_PATH=$EMB_SAVE_PATH/encoded_loc_test_finetune.h5ad
  OUTPUT_DIR="$EXPERIMENT_PATH/rf/loc_raw_quick"

  EMB_NAME=embedding
fi

# -----------------------


SEARCH_TYPE=random
MODEL_TYPE=multiclass
TARGET=location
TARGET_COLNAME=location
SEARCH_TYPE=none

NOTES="rf for $TARGET ($MODEL_TYPE). using raw counts. \n
Command line inputs: \n
--train-embed-path $TRAIN_EMBED_PATH \n
--test-embed-path $TEST_EMBED_PATH \n
--output-dir $OUTPUT_DIR \n
--search-type $SEARCH_TYPE \n
--model-type $MODEL_TYPE \n
--target-colname $TARGET_COLNAME \n
--emb-name $EMB_NAME \n
--downstream-task $TARGET \n
--ignored-labels unknown"


mkdir -p "$OUTPUT_DIR"
echo -e "$NOTES" > "$OUTPUT_DIR/notes.txt"
echo -e "$NOTES"

python -m scripts.random_forest_exp \
  --train-embed-path $TRAIN_EMBED_PATH \
  --test-embed-path $TEST_EMBED_PATH \
  --output-dir $OUTPUT_DIR \
  --search-type $SEARCH_TYPE \
  --model-type $MODEL_TYPE \
  --target-colname $TARGET_COLNAME \
  --emb-name $EMB_NAME \
  --downstream-task $TARGET \
  --ignored-labels unknown

echo "Random Forest job finished."
