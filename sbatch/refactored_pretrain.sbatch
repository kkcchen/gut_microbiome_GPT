#!/bin/bash
#SBATCH --job-name=pretrain_contrastive_refactored
#SBATCH -A aip-rahulgk
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/refactored/pretrain/%j.out
#SBATCH --error=logs/refactored/pretrain/%j.err
#SBATCH --gres=gpu:1

# prepare your environment here
source ~/hmbenv/bin/activate
cd /project/aip-rahulgk/dpellow/gut_microbiome_GPT

# put your command here
DATA_DIR="/project/aip-rahulgk/gutmodel/hmc_final"
RUN_NAME="gat-1"
NOTES="10/24 experiment with gnn encoder, taxa decoder, and mvc task. gnn decoder is gat instead of gcn."

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
    --wandb-project "hmbgpt_gnn" \
    --wandb-run-name "$RUN_NAME" \
    --wandb-run-notes "$NOTES" \
    --init-lr 1e-3 \
    --batch-size 64 \
    --max-epochs 16 \
    --cosine-warmup-ratio-or-step 1000 \
    --num-bins 15 \
    --log-interval 10 \
    --enable-fp16 \
    --seed 42 \
    --notes "$NOTES" \
    --train-mask-ratio 0.4 \
    --do-taxa-decoder \
    --do-mvc \
    --use-gnn \
    # --use-batch-labels \
    # --do-contrastive &
