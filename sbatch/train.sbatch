#!/bin/bash
#SBATCH --job-name=train_hmc
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=t4v1,t4v2,a40,rtx6000 
#SBATCH --qos=normal
#SBATCH --time=15:00:00
#SBATCH --output=logs/train_hmc/%j.out
#SBATCH --error=logs/train_hmc/%j.err
#SBATCH --gres=gpu:4

# prepare your environment here
cd ~/gut_microbiome_GPT
source ~/hmbenv/bin/activate

python -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available. Exiting.'" || exit 1

# put your command here
DATA_DIR="$HOME/datasets"
HMC_TABLE_PATH="$DATA_DIR/pretrain/taxonomy_table_512_train.npy"
TAXA_JSON_PATH="$DATA_DIR/pretrain/pretrain_cols.json"
SAVE_DIR="./model_checkpoints"
BEST_DIR="$SAVE_DIR/best_model"
MODEL_CHECKPOINTS="$SAVE_DIR/current_checkpoints"
DATA_RESTORE_PATH="$SAVE_DIR/data_checkpoint/preempt_data.pt"

python -m wandb login 7559085cf6a63fa75982fe12e737bdd576c4fc62

accelerate launch trainer.py \
    --hmc-table-path "$HMC_TABLE_PATH" \
    --taxa-path "$TAXA_JSON_PATH" \
    --best-dir "$BEST_DIR" \
    --checkpoint-dir "$MODEL_CHECKPOINTS" \
    --data-restore-path "$DATA_RESTORE_PATH" \
    --wandb-enabled \
    --wandb-entity "haoze-deng-university-of-toronto" \
    --wandb-project "hmbgpt_oncluster" \
    --init-lr 1e-3 \
    --batch-size 64 \
    --max-epochs 25 \
    --cosine-warmup-ratio-or-step 100 \
    --num-bins 30 \
    --log-interval 10 \
    --seed 42
