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

EXPERIMENT_PATH=experiment_saves/taxa-mask-30-locexc
EMB_SAVE_PATH=$EXPERIMENT_PATH/encoded_samples

TRAIN_EMBED_PATH=$EMB_SAVE_PATH/encoded_samples_train.h5ad
TEST_EMBED_PATH=$EMB_SAVE_PATH/encoded_samples_test.h5ad

SEARCH_TYPE="random"

for TARGET in sex age bmi mice_diet supplement human_diet
do
    # Pick model type and target column name
    if [[ "$TARGET" == "age" || "$TARGET" == "bmi" ]]; then
        MODEL_TYPE="regression"
        TARGET_COLNAME="continuous_label"
    else
        MODEL_TYPE="multiclass"
        TARGET_COLNAME="categorical_label"
    fi

    OUTPUT_DIR="$EXPERIMENT_PATH/rf/${TARGET}_rf"
    NOTES="rf on raw, $TARGET ($MODEL_TYPE)"

    mkdir -p "$OUTPUT_DIR"
    echo "$NOTES" > "$OUTPUT_DIR/notes.txt"

    echo "Running Random Forest for $TARGET with $MODEL_TYPE and $TARGET_COLNAME..."

    # Build the command
    CMD="python -m scripts.random_forest_exp \
        --train-embed-path \"$TRAIN_EMBED_PATH\" \
        --test-embed-path \"$TEST_EMBED_PATH\" \
        --output-dir \"$OUTPUT_DIR\" \
        --search-type \"$SEARCH_TYPE\" \
        --model-type \"$MODEL_TYPE\" \
        --target-colname \"$TARGET_COLNAME\" \
        --emb-name embedding \
        --downstream-task \"$TARGET\""

    # Add ignored-label for sex only
    if [[ "$TARGET" == "sex" ]]; then
        CMD="$CMD --ignored-label other"
    fi

    # Run the command
    eval $CMD
done


echo "Random Forest job finished."
