#!/bin/bash
#SBATCH --job-name=rawmlp
#SBATCH -A aip-rahulgk
#SBATCH --output=logs/rawmlp/%j.out
#SBATCH --error=logs/rawmlp/%j.err
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=6:00:00              

# Activate your virtual environment
cd /project/aip-rahulgk/kchen13/gutmodel/gut_microbiome_GPT
source ~/hmbenv/bin/activate

FROM_RAW=1
FINETUNE=0


if [ "$FROM_RAW" -eq 1 ]; then
    echo "Using raw embeddings..."
    RUN_TYPE="mlp_raw"

    EXPERIMENT_PATH=experiment_saves/raw_abundances_sep10
    EMB_SAVE_PATH=/project/aip-rahulgk/gutmodel/up_to_date_sep10

    TRAIN_EMBED_PATH=$EMB_SAVE_PATH/finetune_train.h5ad
    TEST_EMBED_PATH=$EMB_SAVE_PATH/finetune_test.h5ad

    OUTPUT_DIR_PARENT="$EXPERIMENT_PATH/mlp_raw"

    EMB_NAME=raw_embedding
else
    echo "Using pre-encoded embeddings..."
    EXPERIMENT_PATH=experiment_saves/taxa-mask-30-locexc-batch
    EMB_SAVE_PATH=$EXPERIMENT_PATH/up_to_date_sep10

    TRAIN_EMBED_PATH=$EMB_SAVE_PATH/encoded_train.h5ad
    TEST_EMBED_PATH=$EMB_SAVE_PATH/encoded_test.h5ad

    if [ "$FINETUNE" -eq 0 ]; then
        RUN_TYPE="mlp_zeroshot"
        OUTPUT_DIR_PARENT="$EXPERIMENT_PATH/mlp_zeroshot"
        EMB_NAME=embedding
    else
        RUN_TYPE="mlp_finetune"
        OUTPUT_DIR_PARENT="$EXPERIMENT_PATH/mlp_finetune"
    fi
fi

OUTPUT_DIR_PARENT="$EXPERIMENT_PATH/$RUN_TYPE"



for TARGET in sex location age bmi mice_diet supplement human_diet
do
    RUN_NAME="${RUN_TYPE}_${TARGET}"
    if [ "$FINETUNE" -eq 0 ] && [ "$FROM_RAW" -eq 0 ]; then
        EMB_NAME="${TARGET}_embedding"
    fi
    # Pick model type and target column name
    if [[ "$TARGET" == "age" || "$TARGET" == "bmi" ]]; then
        MODEL_TYPE="regression"
        TARGET_COLNAME="continuous_label"
    else
        MODEL_TYPE="classification"
        TARGET_COLNAME="categorical_label"
    fi

    OUTPUT_DIR="$OUTPUT_DIR_PARENT/${TARGET}_mlp"
    NOTES="mlp on raw, $TARGET ($MODEL_TYPE)"

    mkdir -p "$OUTPUT_DIR"
    echo "$NOTES" > "$OUTPUT_DIR/notes.txt"

    echo "Running MLP for $TARGET with $MODEL_TYPE and $TARGET_COLNAME..."

    # Build the command
    CMD="python -m scripts.rawmlp_trainer \
        --train-embed-path \"$TRAIN_EMBED_PATH\" \
        --test-embed-path \"$TEST_EMBED_PATH\" \
        --output-dir \"$OUTPUT_DIR\" \
        --model-type \"$MODEL_TYPE\" \
        --target-colname \"$TARGET_COLNAME\" \
        --emb-name $EMB_NAME \
        --downstream-task \"$TARGET\" \
        --wandb-enabled \
        --wandb-entity "haoze-deng-university-of-toronto" \
        --wandb-project "hmbgpt_raw_mlp" \
        --wandb-run-name "$RUN_NAME" \
        --wandb-run-notes \"$NOTES\" "

    # Add ignored-label for specific targets
    if [[ "$TARGET" == "sex" ]]; then
        CMD="$CMD --ignored-label other"
    elif [[ "$TARGET" == "location" ]]; then
        CMD="$CMD --ignored-label unknown"
    fi

    # Run the command
    echo "Executing command: $CMD"
    eval $CMD
done


echo "MLP job finished."
