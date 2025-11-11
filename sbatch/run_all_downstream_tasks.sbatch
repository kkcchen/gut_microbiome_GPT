#!/bin/bash
#SBATCH --job-name=microbiome_downstream
#SBATCH -A aip-rahulgk
#SBATCH --output=logs/downstream_tasks/%j.out
#SBATCH --error=logs/downstream_tasks/%j.err
#SBATCH --cpus-per-task=42
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH --time=3:00:00              

# Activate your virtual environment
source ~/hmbenv/bin/activate
cd /project/aip-rahulgk/kchen13/gutmodel/gut_microbiome_GPT
EXPERIMENT_PATH=experiment_saves/clr-scale-randinit
DS_DATA_PATH=$EXPERIMENT_PATH/remove_redundant_oct24
TRAIN_PATH=$DS_DATA_PATH/encoded_train.h5ad
TEST_PATH=$DS_DATA_PATH/encoded_test.h5ad
CONFIG_PATH=configs/downstream_all_embed_end.json
export SCIPY_ARRAY_API=1
CMD="python -m scripts.run_all_downstream_tasks \
    --train-path $TRAIN_PATH \
    --test-path $TEST_PATH \
    --output-path $EXPERIMENT_PATH/tree_train \
    --tasks $CONFIG_PATH"

eval $CMD
echo "downstream tasks job finished."
