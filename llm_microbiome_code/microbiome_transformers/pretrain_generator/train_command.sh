# Record the start time
start_time=$(date +%s)

CUDA_VISIBLE_DEVICES=0 python begin.py --train_dataset ~/guille/train_encodings_512.npy --test_dataset ~/guille/test_encodings_512.npy --vocab_path ~/guille/vocab_embeddings.npy --output_path /scratch/popeq/Research/Rohan_ensemble/training_outputs/generators/gen --batch_size 32 --layers 10 -e 240 --attn_heads 10 --seq_len 513 --cuda --log_file /scratch/popeq/Research/Rohan_ensemble/training_outputs/logs/gen_train_2.txt

# Record the end time
end_time=$(date +%s)

# Calculate and record the duration
duration=$((end_time - start_time))
echo "Training duration: $duration seconds" >> /scratch/popeq/Research/Rohan_ensemble/training_outputs/logs/gen_train_2.txt