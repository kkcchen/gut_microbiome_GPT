#!/bin/bash
#SBATCH --job-name=finetune_locs
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=t4v1,t4v2,a40,rtx6000 
#SBATCH --qos=m2
#SBATCH --time=1:00:00
#SBATCH --output=logs/finetune_locs/%j.out
#SBATCH --error=logs/finetune_locs/%j.err
#SBATCH --gres=gpu:4

set -e  # <--- Exit immediately on error

# prepare your environment here
cd ~/gut_microbiome_GPT
source ~/hmbenv/bin/activate

python -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available. Exiting.'" || exit 1

# === Set paths ===
RUN_NAME=test_valid_long4
NOTES="test if valid results match test results. dist shift?"
BASEMODEL_PATH=experiment_saves/with_batches
EXPERIMENT_PATH=experiment_saves/finetune/$RUN_NAME
DATASET_PATH=/projects/gutmodel/datasets/finetune

PORT=$((29500 + $SLURM_JOB_ID % 1000))

# === Run the encoding script ===
echo "training on finetune data..."
accelerate launch --main_process_port $PORT -m scripts.trainer_finetune \
    --base-model-config-path $BASEMODEL_PATH/model_checkpoints/model_config.json \
    --base-model-path $BASEMODEL_PATH/model_checkpoints/best_model/model.safetensors \
    --vocab-restore-dir $BASEMODEL_PATH/model_checkpoints/data_checkpoint \
    \
    --train-input $DATASET_PATH/train/filtered_outputs/filtered_data_512.npy \
    --train-loc-labels-path $DATASET_PATH/train/filtered_outputs/filtered_locs.json \
    \
    --best-dir $EXPERIMENT_PATH/best_model \
    --data-restore-dir $EXPERIMENT_PATH/data_checkpoint \
    --batch-restore-dir $EXPERIMENT_PATH/data_checkpoint \
    --model-config-path $EXPERIMENT_PATH/finetune_model_config.json \
    \
    --frozen-max-epochs 10 \
    --unfrozen-max-epochs 10 \
    \
    \
    --batch-size 64 \
    --init-lr 0.001 \
    \
    --notes "$NOTES" \
    --wandb-enabled \
    --wandb-entity haoze-deng-university-of-toronto \
    --wandb-project microbiome-finetune-test \
    --wandb-run-name $RUN_NAME \
    --start-over


accelerate launch --main_process_port $PORT -m scripts.tester_finetune \
    --best-path $EXPERIMENT_PATH/best_model/model.safetensors \
    --model-config-path $EXPERIMENT_PATH/finetune_model_config.json \
    \
    --batch-vocab-path $EXPERIMENT_PATH/data_checkpoint/batch_vocab.json \
    \
    --vocab-path $BASEMODEL_PATH/model_checkpoints/data_checkpoint/vocab.json \
    --vocab-metadata-path $BASEMODEL_PATH/model_checkpoints/data_checkpoint/vocab_metadata.json \
    \
    --npy-path $DATASET_PATH/test/filtered_outputs/filtered_data_512.npy \
    --test-loc-labels-path $DATASET_PATH/test/filtered_outputs/filtered_locs.json \
    \
    --output-path $EXPERIMENT_PATH/finetune_test_outputs/test_outputs.json \

echo "testing complete."

