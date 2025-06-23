#!/bin/bash
#SBATCH --job-name=encode_hmc
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=t4v1,t4v2,a40,rtx6000 
#SBATCH --qos=normal
#SBATCH --time=1:00:00
#SBATCH --output=logs/encode_hmc/%j.out
#SBATCH --error=logs/encode_hmc/%j.err
#SBATCH --gres=gpu:4

# prepare your environment here
cd ~/gut_microbiome_GPT
source ~/hmbenv/bin/activate

python -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available. Exiting.'" || exit 1

# === Set paths ===
BASEMODEL_PATH=experiment_saves/without_batches
EXPERIMENT_PATH=experiment_saves/finetune/first_run
DATASET_PATH=/projects/gutmodel/datasets/finetune

# === Run the encoding script ===
echo "training on finetune data..."
accelerate launch -m scripts.trainer_finetune \
    --base-model-config-path $BASEMODEL_PATH/model_checkpoints/model_config.json \
    --base-model-path $BASEMODEL_PATH/model_checkpoints/best_model/model.safetensors \
    --vocab-restore-dir $BASEMODEL_PATH/model_checkpoints/data_checkpoint \
    \
    --train-input $DATASET_PATH/train/filtered_outputs/filtered_data_512.npy \
    --train-loc-labels-path $DATASET_PATH/train/filtered_outputs/filtered_locs.json \
    \
    --best-dir $EXPERIMENT_PATH/loc_finetune/best_model \
    --checkpoint-dir $EXPERIMENT_PATH/current_checkpoints \
    --data-restore-dir $EXPERIMENT_PATH/data_checkpoint \
    --batch-restore-dir $EXPERIMENT_PATH/data_checkpoint \
    --model-config-path $EXPERIMENT_PATH/finetune_model_config.json \
    \
    --frozen-max-epochs 10 \
    --unfrozen-max-epochs 10 \
    \
    --wandb-enabled \
    --wandb-entity vector-kevin \
    --wandb-project microbiome-finetune \
    --wandb-run-name test_run_001 \
    \
    --notes "first run"


accelerate launch -m scripts.tester_finetune \
    --best-path $EXPERIMENT_PATH/loc_finetune/best_model/model.safetensors \
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
    --output-dir $EXPERIMENT_PATH/finetune_test_outputs/test_outputs.json \

echo "testing complete."

