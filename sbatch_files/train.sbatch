#!/bin/bash
#SBATCH --job-name=train_hmc
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --time=6:00:00
#SBATCH --output=logs/train_hmc/%j.out
#SBATCH --error=logs/train_hmc/%j.err
#SBATCH --gres=gpu:1

# prepare your environment here
cd ~/gut_microbiome_GPT
source ~/hmbenv/bin/activate

# put your command here
DATA_DIR="/home/kchen/microbiome/gut_microbiome_GPT/datasets"
HMC_TABLE_PATH="$DATA_DIR/pretrain/taxonomy_table_512_train.npy"
TAXA_JSON_PATH="$DATA_DIR/pretrain/pretrain_cols.json"
SAVE_DIR="./model_checkpoints"
MODEL_CHECKPOINTS="$SAVE_DIR/preemption_checkpoints/preempt_model.pt"
DATA_RESTORE_PATH="$SAVE_DIR/preemption_checkpoints/preempt_data.pt"

python -m wandb login 7559085cf6a63fa75982fe12e737bdd576c4fc62

python train.py \
    --hmc-table-path "$HMC_TABLE_PATH" \
    --taxa-path "$TAXA_JSON_PATH" \
    --save-dir "$SAVE_DIR" \
    --checkpoint-path "$MODEL_CHECKPOINTS" \
    --data-restore-path "$DATA_RESTORE_PATH" \
    --wandb-enabled \
    --wandb-entity "kevinkaiwen-chen-vector" \
    --wandb-project "hmb_oncluster" \
    --init-lr 1e-3 \
    --batch-size 64 \
    --max-epochs 25 \
    --cosine-warmup-ratio 0.1 \
    --num-bins 30 \
    --log-interval 10 \
    --seed 42
