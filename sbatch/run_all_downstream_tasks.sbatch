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
EXPERIMENT_PATH=experiment_saves/contrastive_downstream_final_arcsine
EMBED_PATH=experiment_saves/contrastive_embed/
TRAIN_PATH=$EMBED_PATH/encoded_contrastive_final_arcsine_train_best.h5ad
TEST_PATH=$EMBED_PATH/encoded_contrastive_final_arcsine_test_best.h5ad
CONFIG_PATH=configs/downstream_all_embed.json
export SCIPY_ARRAY_API=1
CMD="python -m scripts.run_all_downstream_tasks \
    --train-path $TRAIN_PATH \
    --test-path $TEST_PATH \
    --output-path $EXPERIMENT_PATH/tree_train \
    --tasks $CONFIG_PATH"

eval $CMD
echo "downstream tasks job finished."
