# README!!!

* data_utils: contains dataset classes
* llm_microbiome_code: contains original code from Pope el al., for reference only as they are not being used
* models: model definitions, contains generator and discriminator as of now
* notebooks: miscellaneous notebooks I've created when first understanding datasets, disregard
* scripts: scripts used for generating datasets I've processed on OneDrive, read through to understand my thought process at that time
* trainers: contains trainers for generator, discriminator, and finetuning
* inference_electra_discriminator.py: run this with proper args for inference with electra
* train_electra_discriminator.py: does exactly what the name says
* train_electra_generator.py: ^

## sample usage
### training generator
```
python train_electra_generator.py 
    --train_dataset [path here]
    --test_dataset [path here]
    --vocab_path [path here]
    --output_path [path here]
    --batch_size 24
    --layers 10
    --epochs 240
    --attn_heads 10
    --seq_len 513
    --hidden 100
    --cuda
    --log_file [path here]
    --use_wandb True
    --slurm_checkpoint [path here]
    --wandb_savepoint [path here]
    --checkpoint_freq 1 
```

### training discriminator
```
python train_electra_discriminator.py
--train_dataset
/home/kevin/Desktop/gut_microbiome/dataset/llm_microbiome/microbiomedata/train_encodings_512.npy
--test_dataset
/home/kevin/Desktop/gut_microbiome/dataset/llm_microbiome/microbiomedata/test_encodings_512.npy
--vocab_path
/home/kevin/Desktop/gut_microbiome/dataset/llm_microbiome/vocab_embeddings.npy
-o
/home/kevin/Desktop/gut_microbiome/outputs_agp/electra_discriminator
-b
32
-l
5
-a
5
-hs
100
-ghs
100
-e
15
-s
513
-gl
10
-ga
10
-gs
513
--cuda
--d_log_file
/home/kevin/Desktop/gut_microbiome/llm_microbiome_code/microbiome_transformers/outputs/hmc_disc/logs_debug.txt
--load_gen
/home/kevin/Desktop/gut_microbiome/outputs_agp/electra_generator/generator/epoch_29/generator_weights
/home/kevin/Desktop/gut_microbiome/outputs_agp/electra_generator/generator/epoch_89/generator_weights
/home/kevin/Desktop/gut_microbiome/outputs_agp/electra_generator/generator/epoch_149/generator_weights
/home/kevin/Desktop/gut_microbiome/outputs_agp/electra_generator/generator/epoch_209/generator_weights
/home/kevin/Desktop/gut_microbiome/outputs_agp/electra_generator/generator/epoch_219/generator_weights
/home/kevin/Desktop/gut_microbiome/outputs_agp/electra_generator/generator/epoch_229/generator_weights
/home/kevin/Desktop/gut_microbiome/outputs_agp/electra_generator/generator/epoch_239/generator_weights
--load_g_embed
/home/kevin/Desktop/gut_microbiome/outputs_agp/electra_generator/generator/epoch_29/embed_layer_weights
/home/kevin/Desktop/gut_microbiome/outputs_agp/electra_generator/generator/epoch_89/embed_layer_weights
/home/kevin/Desktop/gut_microbiome/outputs_agp/electra_generator/generator/epoch_149/embed_layer_weights
/home/kevin/Desktop/gut_microbiome/outputs_agp/electra_generator/generator/epoch_209/embed_layer_weights
/home/kevin/Desktop/gut_microbiome/outputs_agp/electra_generator/generator/epoch_219/embed_layer_weights
/home/kevin/Desktop/gut_microbiome/outputs_agp/electra_generator/generator/epoch_229/embed_layer_weights
/home/kevin/Desktop/gut_microbiome/outputs_agp/electra_generator/generator/epoch_239/embed_layer_weights
--freeze
--use_wandb
True
--slurm_checkpoint
/home/kevin/Desktop/gut_microbiome/outputs_agp/electra_discriminator/checkpoints
--wandb_savepoint
/home/kevin/Desktop/gut_microbiome/outputs_agp/electra_discriminator_wandb_runid.txt

```
