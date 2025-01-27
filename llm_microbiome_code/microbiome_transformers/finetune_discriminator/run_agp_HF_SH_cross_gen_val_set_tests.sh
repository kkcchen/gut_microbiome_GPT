# HF
run_hf(){
    for i in {1..5}
    do
        CUDA_VISIBLE_DEVICES=1 python begin.py \
            --samples ~/guille/total_IBD_512.npy \
            --sample_labels ~/guille/total_IBD_label.npy \
            --test_samples ~/guille/halfvarson_512_otu.npy \
            --test_labels ~/guille/halfvarson_IBD_labels.npy \
            --vocab_path ~/guille/vocab_embeddings.npy \
            --output_path /scratch/popeq/Research/Rohan_ensemble/training_outputs/IBD/IBD_HF_cross_gen_tests_with_val_split \
            --batch_size 32 \
            --layers 5 \
            --epochs 50 \
            --attn_heads 5 \
            --cuda \
            --log_file /scratch/popeq/Research/Rohan_ensemble/training_outputs/logs/IBD_HF_cross_gen_tests_with_val_split_log_${i}.txt \
            --freeze_opt 2 \
            --freeze_encoders 5 \
            --weighted_sampler \
            -s 513 \
            --num_labels 2 \
            --load_disc /scratch/popeq/Research/Rohan_ensemble/pretrainedmodels/5head5layer_epoch120_disc/ \
            --mse \
            --sgd \
            --lr 0.01 \
            --cross_gen_test \
            --val_split_cross_gen_frac 0.1 \
            --data_split_seed $i
    done
}

# SH
run_sh(){
    for i in {1..5}
        do
        CUDA_VISIBLE_DEVICES=0 python begin.py \
            --samples ~/guille/total_IBD_512.npy \
            --sample_labels ~/guille/total_IBD_label.npy \
            --test_samples ~/guille/schirmer_IBD_512_otu.npy \
            --test_labels ~/guille/schirmer_IBD_labels.npy \
            --vocab_path ~/guille/vocab_embeddings.npy \
            --output_path /scratch/popeq/Research/Rohan_ensemble/training_outputs/IBD/IBD_SH_cross_gen_tests_with_val_split \
            --batch_size 32 \
            --layers 5 \
            --epochs 50 \
            --attn_heads 5 \
            --cuda \
            --log_file /scratch/popeq/Research/Rohan_ensemble/training_outputs/logs/IBD_SH_cross_gen_tests_with_val_split_log_${i}.txt \
            --freeze_opt 2 \
            --freeze_encoders 5 \
            --weighted_sampler \
            -s 513 \
            --num_labels 2 \
            --load_disc /scratch/popeq/Research/Rohan_ensemble/pretrainedmodels/5head5layer_epoch120_disc/ \
            --mse \
            --sgd \
            --lr 0.01 \
            --cross_gen_test \
            --val_split_cross_gen_frac 0.1 \
            --data_split_seed $i
    done
}

if [ "$1" = "HF" ]; then
    echo "Running HF"
    run_hf
elif [ "$1" = "SH" ]; then
    echo "Running SH"
    run_sh
elif [ "$1" = "all" ]; then
    echo "Running all"
    run_hf &
    run_sh &
    wait
else
    echo "Invalid argument. Please use 'HF', 'SH' or 'all'."
    exit 1
fi