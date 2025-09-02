#!/bin/bash
#SBATCH --job-name=encode_hmc
#SBATCH -A aip-rahulgk
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --output=logs/encode_hmc/%j.out
#SBATCH --error=logs/encode_hmc/%j.err
#SBATCH --gres=gpu:4

# prepare your environment here
cd /project/aip-rahulgk/kchen13/gutmodel/gut_microbiome_GPT
source ~/hmbenv/bin/activate

python -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available. Exiting.'" || exit 1

# === Set paths ===
EXPERIMENT_PATH=experiment_saves/taxa-mask-30-locexc-batch

SAFETENSORS_PATH=$EXPERIMENT_PATH/model_checkpoints/best_model/model.safetensors
VOCAB_PATH=$EXPERIMENT_PATH/model_checkpoints/data_checkpoint/augmented_data.h5ad
MODEL_CONFIG_PATH=$EXPERIMENT_PATH/model_checkpoints/model_config.json

TRAIN_ANN_PATH=/project/aip-rahulgk/gutmodel/datasets_newsplit/finetune_train.h5ad
TEST_ANN_PATH=/project/aip-rahulgk/gutmodel/datasets_newsplit/finetune_test.h5ad

TRAIN_OUTPUT_PATH=$EXPERIMENT_PATH/encoded_samples/encoded_train.h5ad
TEST_OUTPUT_PATH=$EXPERIMENT_PATH/encoded_samples/encoded_test.h5ad

# === Run the encoding script ===
echo "Running encoding for finetune train set..."
accelerate launch -m scripts.encode_samples \
  --safetensors-path $SAFETENSORS_PATH \
  --output-path $TRAIN_OUTPUT_PATH \
  --vocab-path $VOCAB_PATH \
  --model-config-path $MODEL_CONFIG_PATH \
  --adata-path $TRAIN_ANN_PATH \
  # --is-finetune \


echo "Running encoding for finetune train set..."
accelerate launch -m scripts.encode_samples \
  --safetensors-path $SAFETENSORS_PATH \
  --output-path $TEST_OUTPUT_PATH \
  --vocab-path $VOCAB_PATH \
  --model-config-path $MODEL_CONFIG_PATH \
  --adata-path $TEST_ANN_PATH \
  # --is-finetune \

echo "Encoding complete."
