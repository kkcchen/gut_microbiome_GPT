#!/bin/bash

accelerate launch scripts/trainer.py \
    --model-config-path model_checkpoints/model_config.json \
    --hmc-table-path datasets/pretrain/taxonomy_table_512.npy \
    --taxa-path datasets/pretrain/pretrain_cols.json \
    --best-dir model_checkpoints/best_model \
    --checkpoint-dir model_checkpoints/current_checkpoints \
    --data-restore-dir model_checkpoints/data_checkpoint \
    --experiments-path datasets/pretrain/sample_list.json \
    --batch-size 1 \
    --log-interval 4 \
    --cosine-warmup-ratio-or-step 0.1 \
    --nrows 100 \
    --seed 1 \
    --num-bins 15 \
    --use-batch-labels \
    --start-over \
    --max-epochs 1 \
    # --wandb-enabled \
    # --wandb-entity kevinkaiwen-chen-vector \
    # --wandb-project hmb_testproject