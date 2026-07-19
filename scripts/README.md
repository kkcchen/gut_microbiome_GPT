# Instructions
## Before You Start
- make sure venv is installed and activated.
- ensure configs are correct in `/configs`, pretrain config can be found in `/configs/pretrain/debug.yaml`, eval config can be found in `/configs/eval/debug.yaml`.
## Pretrain
This is for pretraining the model from scratch with initialization parameters specified in config file. 
from main directory, run 

`accelerate launch --main_process_port 0 -m scripts.train --config <CONFIG_PATH>`

## Eval
This is for obtaining embeddings for samples. 

`accelerate launch --main_process_port 0 -m scripts.eval --config <CONFIG_PATH>`