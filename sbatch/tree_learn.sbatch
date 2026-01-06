#!/bin/bash
#SBATCH --job-name=tree_train
#SBATCH -A aip-rahulgk
#SBATCH --output=logs/tree_train/%j.out
#SBATCH --error=logs/tree_train/%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00              

# Activate your virtual environment
cd /project/aip-rahulgk/kchen13/gutmodel/gut_microbiome_GPT
source ~/hmbenv/bin/activate

FROM_RAW=0
FINETUNE=0

if [ "$FROM_RAW" -eq 1 ]; then
    echo "Using raw embeddings..."
    EXPERIMENT_PATH=experiment_saves/raw_abundances_oct24
    EMB_SAVE_PATH=/project/aip-rahulgk/gutmodel/remove_redundant_oct24

    TRAIN_EMBED_PATH=$EMB_SAVE_PATH/finetune_train.h5ad
    TEST_EMBED_PATH=$EMB_SAVE_PATH/finetune_test.h5ad

    OUTPUT_DIR_PARENT="$EXPERIMENT_PATH/xgb_raw_bigger"

    EMB_NAME=raw_embedding
else
    echo "Using pre-encoded embeddings..."
    EXPERIMENT_PATH=experiment_saves/taxa-mask-30-locexc-batch
    EMB_SAVE_PATH=$EXPERIMENT_PATH/remove_redundant_oct24

    TRAIN_EMBED_PATH=$EMB_SAVE_PATH/encoded_train.h5ad
    TEST_EMBED_PATH=$EMB_SAVE_PATH/encoded_test.h5ad

    if [ "$FINETUNE" -eq 0 ]; then
        OUTPUT_DIR_PARENT="$EXPERIMENT_PATH/xgb_zeroshot"
    else
        OUTPUT_DIR_PARENT="$EXPERIMENT_PATH/xgb_finetune"
    fi
fi

SEARCH_TYPE="none"

for TARGET in sex location age bmi mice_diet supplement human_diet
# for TARGET in location
do
    if [ "$FROM_RAW" -eq 0 ]; then
        if [ "$FINETUNE" -eq 0 ]; then
            EMB_NAME=embedding
        else
            EMB_NAME="${TARGET}_embedding"
        fi
    fi
    # Pick model type and target column name
    if [[ "$TARGET" == "age" || "$TARGET" == "bmi" ]]; then
        MODEL_TYPE="regression"
        TARGET_COLNAME="continuous_label"
    else
        MODEL_TYPE="multiclass"
        TARGET_COLNAME="categorical_label"
    fi

    OUTPUT_DIR="$OUTPUT_DIR_PARENT/${TARGET}_${MODEL_TYPE}"

    mkdir -p "$OUTPUT_DIR"

    echo "Tree model training for $TARGET with $MODEL_TYPE and $TARGET_COLNAME..."

    # Build the command
    CMD="python -m scripts.tree_learn \
        --train-embed-path \"$TRAIN_EMBED_PATH\" \
        --test-embed-path \"$TEST_EMBED_PATH\" \
        --output-dir \"$OUTPUT_DIR\" \
        --search-type \"$SEARCH_TYPE\" \
        --model-arch \"xgb\" \
        --model-type \"$MODEL_TYPE\" \
        --target-colname \"$TARGET_COLNAME\" \
        --emb-name $EMB_NAME \
        --downstream-task \"$TARGET\""

    # Add ignored-label for specific targets
    if [[ "$TARGET" == "sex" ]]; then
        CMD="$CMD --ignored-label other"
    elif [[ "$TARGET" == "location" ]]; then
        CMD="$CMD --ignored-label unknown"
    fi

    echo "$CMD" > "$OUTPUT_DIR/notes.txt"

    # Run the command
    eval $CMD
done


echo "Tree model training job finished."
