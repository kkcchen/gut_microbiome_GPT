#!/bin/bash

accelerate launch -m scripts.trainer \
    --model-config-path experiment_saves/model_checkpoints/model_config.json \
    --hmc-table-path datasets/pretrain/taxonomy_table_512.npy \
    --best-dir experiment_saves/model_checkpoints/best_model \
    --checkpoint-dir experiment_saves/model_checkpoints/current_checkpoints \
    --data-restore-dir experiment_saves/model_checkpoints/data_checkpoint \
    --intermediate-dir experiment_saves/model_checkpoints/intermediate_checkpoints \
    --batch-size 1 \
    --log-interval 4 \
    --cosine-warmup-ratio-or-step 0.1 \
    --nrows 100 \
    --seed 1 \
    --num-bins 15 \
    --use-batch-labels \
    --start-over \
    --max-epochs 1 \
    --do-contrastive \
    # --wandb-enabled \
    # --wandb-entity kevinkaiwen-chen-vector \
    # --wandb-project hmb_testproject \