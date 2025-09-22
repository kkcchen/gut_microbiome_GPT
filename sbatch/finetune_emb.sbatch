#!/bin/bash
#SBATCH --job-name=finetune_trainablebase
#SBATCH -A aip-rahulgk
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --output=logs/finetune_trainablebase/%j.out
#SBATCH --error=logs/finetune_trainablebase/%j.err
#SBATCH --gres=gpu:4

set -e  # <--- Exit immediately on error

# prepare your environment here
cd /project/aip-rahulgk/kchen13/gutmodel/gut_microbiome_GPT
source ~/hmbenv/bin/activate

python -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available. Exiting.'" || exit 1

# === Set paths ===
BASEMODEL_PATH=experiment_saves/taxa-mask-30-locexc-batch
DATASET_PATH=/project/aip-rahulgk/gutmodel/up_to_date_sep10

PORT=$((29500 + $SLURM_JOB_ID % 1000))


# === Run the encoding script ===
# for TARGET in sex location age bmi mice_diet supplement human_diet
for TARGET in location
do
    echo "======================================"
    echo "Running finetune for task: $TARGET"
    echo "======================================"

    # Decide classification vs regression
    if [[ "$TARGET" == "age" || "$TARGET" == "bmi" ]]; then
        TASK_TYPE="regression"
        TARGET_COLNAME="continuous_label"
    else
        TASK_TYPE="classification"
        TARGET_COLNAME="categorical_label"
    fi

    # Ignored labels depending on target
    IGNORED_LABEL=""
    if [[ "$TARGET" == "sex" ]]; then
        IGNORED_LABEL="--ignored-label other"
    elif [[ "$TARGET" == "location" ]]; then
        IGNORED_LABEL="--ignored-label unknown"
    else
        IGNORED_LABEL=""
    fi

    # Experiment path
    RUN_NAME="trainable_${TARGET}2"
    EXPERIMENT_PATH=$BASEMODEL_PATH/finetune_trainablebase/$RUN_NAME
    NOTES="Finetune for $TARGET ($TASK_TYPE)."

    echo "training on finetune data..."
    accelerate launch --main_process_port $PORT -m scripts.trainer_finetune \
        --base-model-config-path $BASEMODEL_PATH/model_checkpoints/model_config.json \
        --base-model-path $BASEMODEL_PATH/model_checkpoints/best_model/model.safetensors \
        --vocab-restore-path $BASEMODEL_PATH/model_checkpoints/data_checkpoint/augmented_data.h5ad \
        \
        --train-input $DATASET_PATH/finetune_train.h5ad \
        \
        --best-dir $EXPERIMENT_PATH/best_model \
        --data-restore-path $EXPERIMENT_PATH/augmented_data_finetune.h5ad \
        \
        --model-config-path $EXPERIMENT_PATH/finetune_model_config.json \
        \
        --frozen-max-epochs 0 \
        --unfrozen-max-epochs 10 \
        \
        --batch-size 64 \
        --init-lr 0.0001 \
        \
        --notes "$NOTES" \
        --wandb-enabled \
        --wandb-entity haoze-deng-university-of-toronto \
        --wandb-project microbiome-finetune-batch \
        --wandb-run-name $RUN_NAME \
        --seed 44 \
        \
        --task-type $TASK_TYPE \
        --downstream-task $TARGET \
        $IGNORED_LABEL \
        --start-over


    accelerate launch --main_process_port $PORT -m scripts.tester_finetune \
        --best-path $EXPERIMENT_PATH/best_model/model.safetensors \
        --model-config-path $EXPERIMENT_PATH/finetune_model_config.json \
        \
        --finetune-vocab-path $EXPERIMENT_PATH/augmented_data_finetune.h5ad \
        \
        --anndata-path $DATASET_PATH/finetune_test.h5ad \
        \
        --output-dir $EXPERIMENT_PATH \
        --task-type $TASK_TYPE \
        --downstream-task $TARGET \
        $IGNORED_LABEL
done

echo "testing complete."

