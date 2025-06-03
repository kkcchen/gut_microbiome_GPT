#!/bin/bash

accelerate launch trainer.py \
    --hmc-table-path datasets/pretrain/taxonomy_table_512_test.npy \
    --taxa-path datasets/pretrain/pretrain_cols.json \
    --best-dir model_checkpoints/best \
    --checkpoint-dir model_checkpoints/current_checkpoints \
    --data-restore-dir model_checkpoints/data/data_state.pt \
    --batch-size 1 \
    --log-interval 4 \
    --cosine-warmup-ratio-or-step 0.1 \
    --nrows 10 \
    --seed 1 \
    --start-over \
    --wandb-enabled \
    --wandb-entity kevinkaiwen-chen-vector \
    --wandb-project hmb_testproject