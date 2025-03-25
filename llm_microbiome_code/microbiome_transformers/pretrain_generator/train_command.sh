# Record the start time
start_time=$(date +%s)

CUDA_VISIBLE_DEVICES=0 python begin.py --train_dataset /home/kevin/Desktop/gut_microbiome/dataset/llm_microbiome/microbiomedata/train_encodings_512.npy --test_dataset /home/kevin/Desktop/gut_microbiome/dataset/llm_microbiome/microbiomedata/test_encodings_512.npy --vocab_path /home/kevin/Desktop/gut_microbiome/dataset/llm_microbiome/vocab_embeddings.npy --output_path /home/kevin/Desktop/gut_microbiome/llm_microbiome_code/microbiome_transformers/outputs --batch_size 32 --layers 10 -e 240 --attn_heads 10 --seq_len 513 --cuda --log_file /home/kevin/Desktop/gut_microbiome/llm_microbiome_code/microbiome_transformers/outputs/logs.txt

# Record the end time
end_time=$(date +%s)

# Calculate and record the duration
duration=$((end_time - start_time))
echo "Training duration: $duration seconds" >> /home/kevin/Desktop/gut_microbiome/llm_microbiome_code/microbiome_transformers/outputs/logs/logs.txt