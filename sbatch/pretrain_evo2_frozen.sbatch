#!/bin/bash
#SBATCH --job-name=gutmodel-evo2init-frozen-normalized
#SBATCH -A aip-rahulgk
#SBATCH --output=logs/evo2-init-frozen-normalized/%j.out
#SBATCH --error=logs/evo2-init-frozen-normalized/%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=1-00:00:00

# prepare your environment here

source ~/hmbenv/bin/activate
cd /project/aip-rahulgk/tiantian/gut_microbiome_GPT

# put your command here
DATA_DIR=/project/aip-rahulgk/gutmodel/up_to_date_sep10
RUN_NAME=evo2-init-frozen-normalized
NOTES="Pretraining pipeline with evo2 embeddings as embedding initialization, frozen during training, 30% masking"

EXPERIMENT_PATH=experiment_saves/$RUN_NAME

ANN_TABLE_PATH=$DATA_DIR/pretrain.h5ad
VOCAB_PATH=$DATA_DIR/hmc_evo2_embeddings.npy

SAVE_DIR=$EXPERIMENT_PATH/model_checkpoints
BEST_DIR=$SAVE_DIR/best_model
MODEL_CHECKPOINTS=$SAVE_DIR/current_checkpoints
DATA_RESTORE_DIR=$SAVE_DIR/data_checkpoint
MODEL_CONFIG_PATH=$SAVE_DIR/model_config.json
INTERMEDIATE_DIR=$SAVE_DIR/intermediate_checkpoints

python -m wandb login 08c24151ddbad507f3c73a88aef41cf3c9acfa5e

accelerate launch --main_process_port 0 -m scripts.trainer \
    --model-config-path "$MODEL_CONFIG_PATH" \
    --ann-table-path "$ANN_TABLE_PATH" \
    --vocab-path "$VOCAB_PATH" \
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
    --max-epochs 25 \
    --cosine-warmup-ratio-or-step 100 \
    --num-bins 15 \
    --log-interval 10 \
    --enable-fp16 \
    --seed 42 \
    --notes "$NOTES" \
    --train-mask-ratio 0.3 \
    --freeze-vocab \
    --use-batch-labels \
    --do-taxa-decoder \
    --do-mvc
    # --do-contrastive &