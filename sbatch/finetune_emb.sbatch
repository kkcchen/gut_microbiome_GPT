#!/bin/bash
#SBATCH --job-name=train_base
#SBATCH -A aip-rahulgk
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --output=logs/train_base/%j.out
#SBATCH --error=logs/train_base/%j.err
#SBATCH --gres=gpu:1

set -e  # <--- Exit immediately on error

# prepare your environment here
cd /home/dpellow/projects/aip-rahulgk/dpellow/gut_microbiome_GPT
source ~/hmbenv/bin/activate

python -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available. Exiting.'" || exit 1

# === Set paths ===
BASEMODEL_PATH=experiment_saves/contrastive_final_projected
DATASET_PATH=/project/aip-rahulgk/gutmodel/hmc_final/

PORT=$((29500 + $SLURM_JOB_ID % 1000))


# === Run the encoding script ===
for TARGET in sex location age bmi crohns_disease supplement diarrhea hbv_infection helicobacter_pylori_infection sunflower_seed_oil_emollient_therapy type_1_diabetes
# for TARGET in location
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
    RUN_NAME="frozen_${TARGET}"
    EXPERIMENT_PATH=$BASEMODEL_PATH/frozen_base/$RUN_NAME
    NOTES="Finetune for $TARGET ($TASK_TYPE)."

    echo "training on finetune data..."
    accelerate launch --main_process_port $PORT -m scripts.trainer_finetune \
        --base-model-config-path $BASEMODEL_PATH/model_checkpoints/model_config.json \
        --base-model-path $BASEMODEL_PATH/model_checkpoints/best_model/model.safetensors \
        --vocab-restore-dir $BASEMODEL_PATH/model_checkpoints/data_checkpoint/ \
        \
        --train-input $DATASET_PATH/downstream_train.h5ad \
        \
        --best-dir $EXPERIMENT_PATH/best_model \
        --data-restore-path $EXPERIMENT_PATH/augmented_data_finetune.h5ad \
        \
        --model-config-path $EXPERIMENT_PATH/finetune_model_config.json \
        \
        --frozen-max-epochs 0 \
        --unfrozen-max-epochs 10 \
        \
        --batch-size 128 \
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
        --vocab-dir $BASEMODEL_PATH/model_checkpoints/data_checkpoint/ \
        \
        --anndata-path $DATASET_PATH/downstream_test.h5ad \
        \
        --output-dir $EXPERIMENT_PATH \
        --task-type $TASK_TYPE \
        --downstream-task $TARGET \
        $IGNORED_LABEL
done

echo "testing complete."

