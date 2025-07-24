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
RUN_NAME=run11-scratch
NOTES="from scratch 12 epochs"
BASEMODEL_PATH=experiment_saves/taxa-mask-real-mvc
EXPERIMENT_PATH=experiment_saves/finetune_tests/run11/$RUN_NAME
DATASET_PATH=/projects/gutmodel/datasets/finetune

PORT=$((29500 + $SLURM_JOB_ID % 1000))

# === Run the encoding script ===
echo "training on finetune data..."
accelerate launch --main_process_port $PORT -m scripts.trainer_finetune \
    --base-model-config-path $BASEMODEL_PATH/model_checkpoints/model_config.json \
    --vocab-restore-dir $BASEMODEL_PATH/model_checkpoints/data_checkpoint \
    \
    --train-input $DATASET_PATH/train/filtered_outputs/filtered_data_512.npy \
    --train-loc-labels-path $DATASET_PATH/train/filtered_outputs/filtered_locs.json \
    \
    --best-dir $EXPERIMENT_PATH/best_model \
    --data-restore-dir experiment_saves/finetune_tests/run11/data_checkpoint \
    --batch-restore-dir experiment_saves/finetune_tests/run11/data_checkpoint \
    \
    --model-config-path $EXPERIMENT_PATH/finetune_model_config.json \
    \
    --frozen-max-epochs 0 \
    --unfrozen-max-epochs 12 \
    \
    \
    --batch-size 64 \
    --init-lr 0.001 \
    \
    --notes "$NOTES" \
    --wandb-enabled \
    --wandb-entity haoze-deng-university-of-toronto \
    --wandb-project microbiome-finetune-epochtest \
    --wandb-run-name $RUN_NAME \
    --seed 43 \
    --task-type classification \
    # --start-over
    # --base-model-path $BASEMODEL_PATH/model_checkpoints/best_model/model.safetensors \

    # --data-restore-dir experiment_saves/finetune_tests/run9/data_checkpoint \
    # --batch-restore-dir experiment_saves/finetune_tests/run9/data_checkpoint \

    # --data-restore-dir $EXPERIMENT_PATH/data_checkpoint \
    # --batch-restore-dir $EXPERIMENT_PATH/data_checkpoint \
    # --base-model-path $BASEMODEL_PATH/model_checkpoints/best_model/model.safetensors \
    # --base-model-path $BASEMODEL_PATH/model_checkpoints/intermediate_checkpoints/epoch_0/model.safetensors \


accelerate launch --main_process_port $PORT -m scripts.tester_finetune \
    --best-path $EXPERIMENT_PATH/best_model/model.safetensors \
    --model-config-path $EXPERIMENT_PATH/finetune_model_config.json \
    \
    --batch-vocab-path experiment_saves/finetune_tests/run11/data_checkpoint/batch_vocab.json \
    \
    --vocab-path $BASEMODEL_PATH/model_checkpoints/data_checkpoint/vocab.json \
    --vocab-metadata-path $BASEMODEL_PATH/model_checkpoints/data_checkpoint/vocab_metadata.json \
    \
    --npy-path $DATASET_PATH/test/filtered_outputs/filtered_data_512.npy \
    --test-loc-labels-path $DATASET_PATH/test/filtered_outputs/filtered_locs.json \
    \
    --output-path $EXPERIMENT_PATH/test_results.json \

    # --batch-vocab-path experiment_saves/finetune_tests/run11/data_checkpoint/data_checkpoint/batch_vocab.json \
    # --batch-vocab-path $EXPERIMENT_PATH/data_checkpoint/batch_vocab.json \

echo "testing complete."

