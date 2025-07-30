source ~/hmbenv/bin/activate

python -m scripts.encode_raw \
  --output-dir experiment_saves/raw_abundances_halfsplit \
  --train-input /project/aip-rahulgk/gutmodel/datasets_halfsplit/finetune/train/finetune_data_train.h5ad \
  --test-input /project/aip-rahulgk/gutmodel/datasets_halfsplit/finetune/test/finetune_data_test.h5ad \
  --prevalence-threshold 0.01 \
  --abundance-threshold 0.05