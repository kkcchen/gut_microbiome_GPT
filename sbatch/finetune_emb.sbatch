#!/bin/bash
#SBATCH --job-name=finetune_emb_test
#SBATCH -A aip-rahulgk
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --output=logs/finetune_emb_test/%j.out
#SBATCH --error=logs/finetune_emb_test/%j.err
#SBATCH --gres=gpu:4

set -e  # <--- Exit immediately on error

# prepare your environment here
cd /project/aip-rahulgk/kchen13/gutmodel/gut_microbiome_GPT
source ~/hmbenv/bin/activate

python -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available. Exiting.'" || exit 1

# === Set paths ===
RUN_NAME=test-human_diet
NOTES="human_diet test for classification script."
BASEMODEL_PATH=experiment_saves/taxa-mask-30-locexc-batch
EXPERIMENT_PATH=$BASEMODEL_PATH/finetune/$RUN_NAME
DATASET_PATH=/project/aip-rahulgk/gutmodel/datasets_newsplit

PORT=$((29500 + $SLURM_JOB_ID % 1000))

# === Run the encoding script ===
echo "training on finetune data..."
accelerate launch --main_process_port $PORT -m scripts.trainer_finetune \
    --base-model-config-path $BASEMODEL_PATH/model_checkpoints/model_config.json \
    --base-model-path $BASEMODEL_PATH/model_checkpoints/best_model/model.safetensors \
    --vocab-restore-path $BASEMODEL_PATH/model_checkpoints/data_checkpoint/augmented_data.h5ad \
    \
    --train-input $DATASET_PATH/finetune_train.h5ad \
    \
    --best-dir $EXPERIMENT_PATH/best_model \
    --data-restore-dir experiment_saves/finetune_newsplit/data_checkpoint \
    \
    --model-config-path $EXPERIMENT_PATH/finetune_model_config.json \
    \
    --frozen-max-epochs 0 \
    --unfrozen-max-epochs 15 \
    \
    --batch-size 128 \
    --init-lr 0.0001 \
    \
    --notes "$NOTES" \
    --wandb-enabled \
    --wandb-entity haoze-deng-university-of-toronto \
    --wandb-project microbiome-finetune-batch \
    --wandb-run-name $RUN_NAME \
    --seed 42 \
    \
    --task-type classification \
    --downstream-task human_diet \
    --ignored-labels other \
    --start-over


accelerate launch --main_process_port $PORT -m scripts.tester_finetune \
    --best-path $EXPERIMENT_PATH/best_model/model.safetensors \
    --model-config-path $EXPERIMENT_PATH/finetune_model_config.json \
    \
    --finetune-vocab-path experiment_saves/finetune_newsplit/data_checkpoint/augmented_data_finetune.h5ad \
    \
    --anndata-path $DATASET_PATH/finetune_test.h5ad \
    \
    --output-dir $EXPERIMENT_PATH \
    --task-type classification \
    --downstream-task human_diet \
    --ignored-labels other \

echo "testing complete."

