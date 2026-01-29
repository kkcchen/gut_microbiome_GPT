#!/bin/bash
#SBATCH --job-name=gutmodel-contrastive-batched-no_clr
#SBATCH -A aip-rahulgk
#SBATCH --output=logs/contrastive/%j.out
#SBATCH --error=logs/sbatch/contrastive/%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=1-00:00:00

# prepare your environment here

source ~/hmbenv/bin/activate
cd /project/aip-rahulgk/dpellow/gut_microbiome_GPT

# put your command here
DATA_DIR=/project/aip-rahulgk/gutmodel/hmc_final/
RUN_NAME=contrastive_final_arcsine
NOTES="Pretraining pipeline. Uses contrastive on projection from cell_emb, arcsine transform on relative abunance with pseudo-counts."

EXPERIMENT_PATH=experiment_saves/$RUN_NAME

ANN_TABLE_PATH=$DATA_DIR/pretrain.h5ad
# VOCAB_PATH=$DATA_DIR/hmc_evo2_embeddings.npy

SAVE_DIR=$EXPERIMENT_PATH/model_checkpoints
BEST_DIR=$SAVE_DIR/best_model
MODEL_CHECKPOINTS=$SAVE_DIR/current_checkpoints
DATA_RESTORE_DIR=$SAVE_DIR/data_checkpoint
MODEL_CONFIG_PATH=$SAVE_DIR/model_config.json
INTERMEDIATE_DIR=$SAVE_DIR/intermediate_checkpoints

python -m wandb login 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

accelerate launch --main_process_port 0 -m scripts.trainer \
    --model-config-path "$MODEL_CONFIG_PATH" \
    --ann-table-path "$ANN_TABLE_PATH" \
    --best-dir "$BEST_DIR" \
    --checkpoint-dir "$MODEL_CHECKPOINTS" \
    --data-restore-dir "$DATA_RESTORE_DIR" \
    --intermediate-dir "$INTERMEDIATE_DIR" \
    --wandb-enabled \
    --wandb-entity "haoze-deng-university-of-toronto" \
    --wandb-project "hmbgpt_contrastive" \
    --wandb-run-name "$RUN_NAME" \
    --wandb-run-notes "$NOTES" \
    --init-lr 1e-3 \
    --batch-size 128 \
    --max-epochs 50 \
    --cosine-warmup-ratio-or-step 100 \
    --num-bins 15 \
    --log-interval 10 \
    --enable-fp16 \
    --seed 42 \
    --notes "$NOTES" \
    --train-mask-ratio 0.3 \
    --use-batch-labels \
    --do-subsample \
    --do-contrastive \
    --data-bin-strategy "none"