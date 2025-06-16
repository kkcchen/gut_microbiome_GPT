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
EXPERIMENT_PATH=experiment_saves/no_mvc

SAFETENSORS_PATH=$EXPERIMENT_PATH/model_checkpoints/best_model/model.safetensors
VOCAB_PATH=$EXPERIMENT_PATH/model_checkpoints/data_checkpoint/vocab.json
VOCAB_METADATA_PATH=$EXPERIMENT_PATH/model_checkpoints/data_checkpoint/vocab_metadata.json
MODEL_CONFIG_PATH=$EXPERIMENT_PATH/model_checkpoints/model_config.json

TRAIN_NPY_PATH=datasets/finetune/train/finetune_data_train_512.npy
TEST_NPY_PATH=datasets/finetune/test/finetune_data_test_512.npy

TRAIN_OUTPUT_PATH=$EXPERIMENT_PATH/encoded_samples/encoded_samples_train.npy
TEST_OUTPUT_PATH=$EXPERIMENT_PATH/encoded_samples/encoded_samples_test.npy

# === Run the encoding script ===
echo "Running encoding for finetune train set..."
accelerate launch -m scripts.encode_samples \
  --safetensors-path $SAFETENSORS_PATH \
  --output-path $TRAIN_OUTPUT_PATH \
  --vocab-path $VOCAB_PATH \
  --vocab-metadata-path $VOCAB_METADATA_PATH \
  --model-config-path $MODEL_CONFIG_PATH \
  --npy-path $TRAIN_NPY_PATH \


echo "Running encoding for finetune train set..."
accelerate launch -m scripts.encode_samples \
  --safetensors-path $SAFETENSORS_PATH \
  --output-path $TEST_OUTPUT_PATH \
  --vocab-path $VOCAB_PATH \
  --vocab-metadata-path $VOCAB_METADATA_PATH \
  --model-config-path $MODEL_CONFIG_PATH \
  --npy-path $TEST_NPY_PATH \

echo "Encoding complete."
