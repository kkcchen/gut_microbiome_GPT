#!/bin/bash
#SBATCH --job-name=train_hmc
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=t4v1,t4v2,a40,rtx6000 
#SBATCH --qos=normal
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_hmc/%j.out
#SBATCH --error=logs/train_hmc/%j.err
#SBATCH --gres=gpu:4

# prepare your environment here
cd ~/gut_microbiome_GPT
source ~/hmbenv/bin/activate

python -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available. Exiting.'" || exit 1

# put your command here
# DATA_DIR="$HOME/datasets"
RUN_NAME=no_mvc
EXPERIMENT_PATH=experiment_saves/$RUN_NAME

HMC_TABLE_PATH="./datasets/pretrain/taxonomy_table_512.npy"
TAXA_JSON_PATH="./datasets/pretrain/pretrain_cols.json"
STUDY_PATH="./datasets/pretrain/sample_list.json"

SAVE_DIR="$EXPERIMENT_PATH/model_checkpoints"
BEST_DIR="$SAVE_DIR/best_model"
MODEL_CHECKPOINTS="$SAVE_DIR/current_checkpoints"
DATA_RESTORE_DIR="$SAVE_DIR/data_checkpoint"
MODEL_CONFIG_PATH="$SAVE_DIR/model_config.json"

NOTES="removing mvc and batches"

python -m wandb login 7559085cf6a63fa75982fe12e737bdd576c4fc62

accelerate launch -m scripts.trainer \
    --hmc-table-path "$HMC_TABLE_PATH" \
    --taxa-path "$TAXA_JSON_PATH" \
    --best-dir "$BEST_DIR" \
    --checkpoint-dir "$MODEL_CHECKPOINTS" \
    --data-restore-dir "$DATA_RESTORE_DIR" \
    --model-config-path "$MODEL_CONFIG_PATH" \
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
    # --use-batch-labels \
    # --samplename-path "$STUDY_PATH" \
    # --do-mvc

