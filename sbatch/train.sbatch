#!/bin/bash
#SBATCH --job-name=train_hmc
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=t4v1,t4v2,a40,rtx6000 
#SBATCH --qos=m
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_hmc/%j.out
#SBATCH --error=logs/train_hmc/%j.err
#SBATCH --gres=gpu:4

# send this batch script a SIGUSR1 240 seconds
# before we hit our time limit
#SBATCH --signal=B:USR1@240

# Kill training process and resubmit job if it receives a SIGUSR1
handle_timeout_or_preemption() {
  date +"%Y-%m-%d %T"
  echo "Caught timeout or preemption signal"
  echo "Sending SIGINT to child process"
  scancel $SLURM_JOB_ID --signal=SIGINT
  wait $child_pid
  echo "Job step terminated gracefully"
  echo $(date +"%Y-%m-%d %T") "Resubmitting job"
  scontrol requeue $SLURM_JOB_ID
  exit 0
}
trap handle_timeout_or_preemption SIGUSR1

# prepare your environment here
cd ~/gut_microbiome_GPT
source ~/hmbenv/bin/activate

python -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available. Exiting.'" || exit 1

# put your command here
DATA_DIR="/projects/gutmodel/datasets"
RUN_NAME=no-cellemb-pass-fair
NOTES="remove second step of forward pretrain pass where cell embs are passed back to model for loss_gen. also doubled loss for normal pcpt for fair comparison"

EXPERIMENT_PATH=experiment_saves/$RUN_NAME

HMC_TABLE_PATH="$DATA_DIR/pretrain/taxonomy_table_512.npy"
TAXA_JSON_PATH="$DATA_DIR/pretrain/pretrain_cols.json"
STUDY_PATH="$DATA_DIR/pretrain/sample_list.json"

SAVE_DIR="$EXPERIMENT_PATH/model_checkpoints"
BEST_DIR="$SAVE_DIR/best_model"
MODEL_CHECKPOINTS="$SAVE_DIR/current_checkpoints"
DATA_RESTORE_DIR="$SAVE_DIR/data_checkpoint"
MODEL_CONFIG_PATH="$SAVE_DIR/model_config.json"


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
    --do-mvc \
    --do-contrastive &
    # --use-batch-labels \
    # --samplename-path "$STUDY_PATH" \

child_pid=$!
wait $child_pid