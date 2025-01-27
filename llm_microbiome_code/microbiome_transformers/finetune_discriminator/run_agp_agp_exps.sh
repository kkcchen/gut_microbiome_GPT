# IBD
run_ibd(){
    for i in {1..5}
    do
        CUDA_VISIBLE_DEVICES=0 python begin.py \
            --samples ~/guille/total_IBD_512.npy \
            --sample_labels ~/guille/total_IBD_label.npy \
            --vocab_path ~/guille/vocab_embeddings.npy \
            --output_path /scratch/popeq/Research/Rohan_ensemble/training_outputs/IBD/IBD_cross_validation \
            --batch_size 32 \
            --layers 5 \
            --epochs 50 \
            --attn_heads 5 \
            --cuda \
            --log_file /scratch/popeq/Research/Rohan_ensemble/training_outputs/logs/IBD_blocked_val_no_val_weighted_static_embeddings_log_${i}.txt \
            --freeze_opt 2 \
            --freeze_encoders 5 \
            --weighted_sampler \
            -s 513 \
            --num_labels 2 \
            --load_disc /scratch/popeq/Research/Rohan_ensemble/pretrainedmodels/5head5layer_epoch120_disc/ \
            --mse \
            --sgd \
            --lr 0.01 \
            --path_to_hosts_mapping /scratch/popeq/Research/Rohan_ensemble/microbiome_transformers/finetune_discriminator/host_to_rows_dir/host_to_indices_total_IBD_512.pkl \
            --ensemble_repeat 0 \
            --data_split_seed $i \
            --val_then_no_val_run \
            --use_static_embeddings

    done
}

# Fruit
run_fruit(){
    for i in {1..5}
        do
        CUDA_VISIBLE_DEVICES=1 python begin.py \
            --samples ../../microbiomedata/fruitdata/FRUIT_FREQUENCY_otu_512.npy \
            --sample_labels ../../microbiomedata/fruitdata/FRUIT_FREQUENCY_binary34_labels.npy \
            --vocab_path ~/guille/vocab_embeddings.npy \
            --output_path /scratch/popeq/Research/Rohan_ensemble/training_outputs/Fruit/Fruit_cross_validation \
            --batch_size 32 \
            --layers 5 \
            --epochs 50 \
            --attn_heads 5 \
            --cuda \
            --log_file /scratch/popeq/Research/Rohan_ensemble/training_outputs/logs/Fruit_blocked_val_no_val_weighted_static_embeddings_log_${i}.txt \
            --freeze_opt 2 \
            --freeze_encoders 5 \
            --weighted_sampler \
            -s 513 \
            --num_labels 2 \
            --load_disc /scratch/popeq/Research/Rohan_ensemble/pretrainedmodels/5head5layer_epoch120_disc/ \
            --mse \
            --sgd \
            --lr 0.01 \
            --path_to_hosts_mapping /scratch/popeq/Research/Rohan_ensemble/microbiome_transformers/finetune_discriminator/host_to_rows_dir/host_to_indices_FRUIT_FREQUENCY_512.pkl \
            --ensemble_repeat 0 \
            --data_split_seed $i \
            --val_then_no_val_run \
            --use_static_embeddings
    done
}

# Veg
run_veg(){
    for i in {1..5}
        do
        CUDA_VISIBLE_DEVICES=2 python begin.py \
            --samples ../../microbiomedata/vegdata/VEGETABLE_FREQUENCY_otu_512.npy \
            --sample_labels ../../microbiomedata/vegdata/VEGETABLE_FREQUENCY_binary34_labels.npy \
            --vocab_path ~/guille/vocab_embeddings.npy \
            --output_path /scratch/popeq/Research/Rohan_ensemble/training_outputs/Veg/Veg_cross_validation \
            --batch_size 32 \
            --layers 5 \
            --epochs 50 \
            --attn_heads 5 \
            --cuda \
            --log_file /scratch/popeq/Research/Rohan_ensemble/training_outputs/logs/Veg_blocked_val_no_val_weighted_static_embeddings_log_${i}.txt \
            --freeze_opt 2 \
            --freeze_encoders 5 \
            --weighted_sampler \
            -s 513 \
            --num_labels 2 \
            --load_disc /scratch/popeq/Research/Rohan_ensemble/pretrainedmodels/5head5layer_epoch120_disc/ \
            --mse \
            --sgd \
            --lr 0.01 \
            --path_to_hosts_mapping /scratch/popeq/Research/Rohan_ensemble/microbiome_transformers/finetune_discriminator/host_to_rows_dir/host_to_indices_VEGETABLE_FREQUENCY_512.pkl \
            --ensemble_repeat 0 \
            --data_split_seed $i \
            --val_then_no_val_run \
            --use_static_embeddings
    done
}

if [ "$1" = "ibd" ]; then
    echo "Running IBD"
    run_ibd
elif [ "$1" = "fruit" ]; then
    echo "Running Fruit"
    run_fruit
elif [ "$1" = "veg" ]; then
    echo "Running Veg"
    run_veg
elif [ "$1" = "all" ]; then
    echo "Running all"
    run_ibd &
    run_fruit &
    run_veg &
    wait
else
    echo "Invalid argument. Please use 'ibd', 'fruit', 'veg', or 'all'."
    exit 1
fi





