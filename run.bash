#!/bin/bash

python trainer.py \
    --hmc_table_path datasets/pretrain/taxonomy_table_512_test.npy \
    --taxa_path datasets/pretrain/pretrain_cols.json \
    --save_dir model_checkpoints \
    --checkpoint_path model_checkpoints/current_checkpoints/preempt_model.pt \
    --data_restore_path model_checkpoints/data/data_state.pt \
    --batch_size 1 \
    --log_interval 4 \
    --cosine_warmup_ratio 0.1 \
    --nrows 10 \
    --seed 1 \
    --start_over \
    --wandb_enabled \
    --wandb_entity kevinkaiwen-chen-vector \
    --wandb_project hmb_testproject