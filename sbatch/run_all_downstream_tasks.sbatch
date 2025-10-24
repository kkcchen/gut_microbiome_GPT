#!/bin/bash
#SBATCH --job-name=microbiome_downstream
#SBATCH -A aip-rahulgk
#SBATCH --output=logs/downstream_tasks/%j.out
#SBATCH --error=logs/downstream_tasks/%j.err
#SBATCH --cpus-per-task=42
#SBATCH --mem=64G
#SBATCH --time=8:00:00              

# Activate your virtual environment
source ~/hmbenv/bin/activate
cd /project/aip-rahulgk/tiantian/gut_microbiome_GPT
EXPERIMENT_PATH=experiment_saves/downstream_all_oct24
DS_DATA_PATH=/project/aip-rahulgk/gutmodel/remove_redundant_oct24
TRAIN_PATH=$DS_DATA_PATH/finetune_train.h5ad
TEST_PATH=$DS_DATA_PATH/finetune_test.h5ad
CONFIG_PATH=configs/downstream_all.json

CMD="python -m scripts.run_all_downstream_tasks \
    --train-path $TRAIN_PATH \
    --test-path $TEST_PATH \
    --output-path $EXPERIMENT_PATH \
    --tasks $CONFIG_PATH"

eval $CMD
echo "downstream tasks job finished."
