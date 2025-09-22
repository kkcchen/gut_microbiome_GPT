#!/bin/bash
#SBATCH --job-name=encode_finetune
#SBATCH -A aip-rahulgk
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --output=logs/encode_finetune/%j.out
#SBATCH --error=logs/encode_finetune/%j.err
#SBATCH --gres=gpu:4

cd /project/aip-rahulgk/kchen13/gutmodel/gut_microbiome_GPT
source ~/hmbenv/bin/activate

python -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available. Exiting.'" || exit 1

EXPERIMENT_PATH=experiment_saves/taxa-mask-30-locexc-batch

# Just the category names
CATEGORIES=(
  # age
  bmi
  human_diet
  location
  mice_diet
  sex
  supplement
)

for CATEGORY in "${CATEGORIES[@]}"; do
  FINETUNE="finetune_${CATEGORY}"
  EMB_COLNAME="${CATEGORY}_embedding"

  echo "=== Processing $FINETUNE → $EMB_COLNAME ==="

  SAFETENSORS_PATH=$EXPERIMENT_PATH/finetune_trainablebase/$FINETUNE/best_model/model.safetensors
  VOCAB_PATH=$EXPERIMENT_PATH/model_checkpoints/data_checkpoint/augmented_data.h5ad
  MODEL_CONFIG_PATH=$EXPERIMENT_PATH/finetune_trainablebase/$FINETUNE/finetune_model_config.json

  TRAIN_OUTPUT_PATH=$EXPERIMENT_PATH/up_to_date_sep10/encoded_train.h5ad
  TEST_OUTPUT_PATH=$EXPERIMENT_PATH/up_to_date_sep10/encoded_test.h5ad

  TRAIN_ANN_PATH=$EXPERIMENT_PATH/up_to_date_sep10/encoded_train.h5ad
  TEST_ANN_PATH=$EXPERIMENT_PATH/up_to_date_sep10/encoded_test.h5ad

  echo "Running encoding for train set ($CATEGORY)..."
  accelerate launch -m scripts.encode_samples \
    --safetensors-path $SAFETENSORS_PATH \
    --output-path $TRAIN_OUTPUT_PATH \
    --vocab-path $VOCAB_PATH \
    --model-config-path $MODEL_CONFIG_PATH \
    --adata-path $TRAIN_ANN_PATH \
    --emb-colname $EMB_COLNAME \
    --is-finetune

  echo "Running encoding for test set ($CATEGORY)..."
  accelerate launch -m scripts.encode_samples \
    --safetensors-path $SAFETENSORS_PATH \
    --output-path $TEST_OUTPUT_PATH \
    --vocab-path $VOCAB_PATH \
    --model-config-path $MODEL_CONFIG_PATH \
    --adata-path $TEST_ANN_PATH \
    --emb-colname $EMB_COLNAME \
    --is-finetune
done

echo "All encodings complete."
