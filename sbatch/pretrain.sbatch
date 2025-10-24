#!/bin/bash
#SBATCH --job-name=train_hmc
#SBATCH -A aip-rahulgk
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_hmc/%j.out
#SBATCH --error=logs/train_hmc/%j.err
#SBATCH --gres=gpu:4

# prepare your environment here
cd /project/aip-rahulgk/kchen13/gutmodel/gut_microbiome_GPT
source ~/hmbenv/bin/activate

python -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available. Exiting.'" || exit 1

# put your command here
DATA_DIR="/project/aip-rahulgk/gutmodel/remove_redundant_oct24"
RUN_NAME="big-model"
NOTES="oct 4 bigger model"

EXPERIMENT_PATH=experiment_saves/$RUN_NAME

ANN_TABLE_PATH="$DATA_DIR/pretrain.h5ad"

SAVE_DIR="$EXPERIMENT_PATH/model_checkpoints"
BEST_DIR="$SAVE_DIR/best_model"
MODEL_CHECKPOINTS="$SAVE_DIR/current_checkpoints"
DATA_RESTORE_DIR="$SAVE_DIR/data_checkpoint"
MODEL_CONFIG_PATH="$SAVE_DIR/model_config.json"
INTERMEDIATE_DIR="$SAVE_DIR/intermediate_checkpoints"

python -m wandb login 7559085cf6a63fa75982fe12e737bdd576c4fc62

accelerate launch --main_process_port 0 -m scripts.trainer \
    --model-config-path "$MODEL_CONFIG_PATH" \
    --ann-table-path "$ANN_TABLE_PATH" \
    --best-dir "$BEST_DIR" \
    --checkpoint-dir "$MODEL_CHECKPOINTS" \
    --data-restore-dir "$DATA_RESTORE_DIR" \
    --intermediate-dir "$INTERMEDIATE_DIR" \
    --wandb-enabled \
    --wandb-entity "haoze-deng-university-of-toronto" \
    --wandb-project "hmbgpt_oncluster" \
    --wandb-run-name "$RUN_NAME" \
    --wandb-run-notes "$NOTES" \
    --init-lr 1e-3 \
    --batch-size 64 \
    --max-epochs 16 \
    --cosine-warmup-ratio-or-step 1000 \
    --num-bins 20 \
    --log-interval 10 \
    --enable-fp16 \
    --seed 42 \
    --notes "$NOTES" \
    --train-mask-ratio 0.3 \
    --do-taxa-decoder \
    --do-mvc \
    # --use-batch-labels \
    # --do-contrastive &
