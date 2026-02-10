#!/bin/bash
#SBATCH --job-name=microbiome_downstream
#SBATCH -A aip-rahulgk
#SBATCH --output=logs/downstream_tasks/%j.out
#SBATCH --error=logs/downstream_tasks/%j.err
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH --time=14:00:00              

# Activate your virtual environment
source ~/hmbenv/bin/activate
cd /home/dpellow/projects/aip-rahulgk/dpellow/gut_microbiome_GPT
EXPERIMENT_PATH=outputs/eval/denoising_dm_nll_binomial_raw
EMBED_PATH=outputs/eval/denoising_dm_nll_binomial_raw
TRAIN_PATH=$EMBED_PATH/downstream_train.h5ad
TEST_PATH=$EMBED_PATH/downstream_test.h5ad
CONFIG_PATH=configs/downstream_all_embed.json
export SCIPY_ARRAY_API=1
CMD="python -m scripts.run_all_downstream_tasks \
    --train-path $TRAIN_PATH \
    --test-path $TEST_PATH \
    --output-path $EXPERIMENT_PATH/tree_train \
    --tasks $CONFIG_PATH"

eval $CMD
echo "downstream tasks job finished."
