python -m scripts.encode_raw \
  --output-dir /h/chenke83/gut_microbiome_GPT/experiment_saves/encoded_samples \
  --train-input /h/chenke83/gut_microbiome_GPT/datasets/finetune/train/finetune_data_train.npy \
  --test-input /h/chenke83/gut_microbiome_GPT/datasets/finetune/test/finetune_data_test.npy \
  --prevalence-threshold 0.01 \
  --abundance-threshold 0.05