#!/bin/bash
#SBATCH --job-name=encode_contrastive
#SBATCH -A aip-rahulgk
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=0:10:00
#SBATCH --output=logs/encode_hmc/%j.out
#SBATCH --error=logs/encode_hmc/%j.err
#SBATCH --gres=gpu:l40s:1

# prepare your environment here
cd /project/aip-rahulgk/dpellow/gut_microbiome_GPT

source ~/hmbenv/bin/activate

python -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available. Exiting.'" || exit 1

# === Set paths ===
EXPERIMENT_PATH=experiment_saves/contrastive_final_depth_aux

SAFETENSORS_PATH=$EXPERIMENT_PATH/model_checkpoints/best_model/model.safetensors
# SAFETENSORS_PATH=$EXPERIMENT_PATH/model_checkpoints/intermediate_checkpoints/epoch_3/model.safetensors
VOCAB_DIR=$EXPERIMENT_PATH/model_checkpoints/data_checkpoint
MODEL_CONFIG_PATH=$EXPERIMENT_PATH/model_checkpoints/model_config.json

TRAIN_OUTPUT_PATH=experiment_saves/contrastive_embed/encoded_contrastive_final_depth_aux_train_best.h5ad
TEST_OUTPUT_PATH=experiment_saves/contrastive_embed/encoded_contrastive_final_depth_aux_test_best.h5ad

TRAIN_ANN_PATH=/project/aip-rahulgk/gutmodel/hmc_final/downstream_train.h5ad
TEST_ANN_PATH=/project/aip-rahulgk/gutmodel/hmc_final/downstream_test.h5ad

# === Run the encoding script ===
echo "Running encoding for finetune train set..."
accelerate launch -m scripts.encode_samples \
  --safetensors-path $SAFETENSORS_PATH \
  --output-path $TRAIN_OUTPUT_PATH \
  --vocab-dir $VOCAB_DIR \
  --model-config-path $MODEL_CONFIG_PATH \
  --adata-path $TRAIN_ANN_PATH \


echo "Running encoding for finetune test set..."
accelerate launch -m scripts.encode_samples \
  --safetensors-path $SAFETENSORS_PATH \
  --output-path $TEST_OUTPUT_PATH \
  --vocab-dir $VOCAB_DIR \
  --model-config-path $MODEL_CONFIG_PATH \
  --adata-path $TEST_ANN_PATH \


echo "Encoding complete."
