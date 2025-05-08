import argparse
import pathlib
from typing import Optional, Dict

from torch.utils.data import DataLoader

import torch
from trainers import GeneratorTrainer
from data_utils import ELECTRADataset
from transformers import ElectraConfig
from models import ElectraGenerator
import wandb
import os
import logging
import numpy as np


# def _setup_logger():
#     logger = logging.getLogger(__name__)
#     if not logger.handlers:
#         handler = logging.StreamHandler()
#         formatter = logging.Formatter(
#             "[%(asctime)s] %(levelname)s:%(name)s: %(message)s"
#         )
#         handler.setFormatter(formatter)
#         logger.addHandler(handler)
#     logger.setLevel(logging.INFO)
#     return logger


def init_or_resume_wandb_run(wandb_id_file_path: pathlib.Path,
                             project_name: Optional[str] = None,
                             run_name: Optional[str] = None,
                             config: Optional[Dict] = None):
    """Detect the run id if it exists and resume
        from there, otherwise write the run id to file.

        Returns the config, if it's not None it will also update it first

        NOTE:
            Make sure that wandb_id_file_path.parent exists before calling this function
    """
    # if the run_id was previously saved, resume from there
    if wandb_id_file_path.exists():
        resume_id = wandb_id_file_path.read_text()
        run = wandb.init(
            project=project_name,
            name=run_name,
            id=resume_id,
            config=config or {},
        )
    else:
        # if the run_id doesn't exist, then create a new run
        # and write the run id the file
        run = wandb.init(
            project=project_name,
            name=run_name,
            resume="allow",
            config=config or {},
        )
        wandb_id_file_path.write_text(str(run.id))

    wandb_config = wandb.config
    if config is not None:
        # update the current passed in config with the wandb_config
        config.update(wandb_config)

    return run


def init_or_load_generator(checkpoint_folder_path: pathlib.Path,
                           vocab: np.array,
                           model_config: ElectraConfig):
    """Detect the previous checkpointing for slurm
        checkpoint_folder should contain lastest model as pt and a gen_embed as pt
        also resume from the appropriate epoch count
        Returns ElectraTrainer, with appropripate resume of training
    """
    # Paths inside the folder
    gen_embed_path = checkpoint_folder_path / "gen_embed"
    ckpt_path = checkpoint_folder_path / "gen_model"
    resume_path = checkpoint_folder_path / "resume_from.txt"

    can_resume = (
            gen_embed_path.is_file() and
            ckpt_path.is_dir() and
            resume_path.is_file()
    )

    if can_resume:
        # Load the epoch to resume from
        resume_epoch = int(resume_path.read_text().strip())

        # Load embeddings and model
        model = ElectraGenerator(model_config, torch.from_numpy(vocab), ckpt_path,
                                 gen_embed_path)
        print(f"Resuming from epoch {resume_epoch}")
    else:
        # Fresh start
        resume_epoch = 0
        model = ElectraGenerator(model_config, torch.from_numpy(vocab), generator=None, embed_layer=None)
        print("No checkpoint found, starting training from beginning")

    return model, resume_epoch


def train():
    """
    Main function to set up and run the ELECTRA model training process.

    This function performs the following steps:
    1. Parses command-line arguments for training configuration.
    2. Loads training and optional test datasets.
    3. Creates data loaders for the datasets.
    4. Initializes the ELECTRA model configuration for the generator.
    5. Sets up the ELECTRA trainer with specified parameters.
    6. Runs the training loop for the specified number of epochs.
    7. Saves the model periodically and after training.
    8. Evaluates the model on the test dataset if provided.

    The function uses argparse to handle various command-line options for
    customizing the training process, including dataset paths, model architecture,
    training hyperparameters, and hardware settings.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--train_dataset", required=True, type=str, help="train dataset for train electra")
    parser.add_argument("-t", "--test_dataset", type=str, default=None, help="test set for evaluate train set")
    parser.add_argument("-v", "--vocab_path", required=True, type=str, help="built vocab model path with electra-vocab")
    parser.add_argument("-o", "--output_path", required=True, type=str, help="ex)output/firstmodel")

    parser.add_argument("-hs", "--hidden", type=int, default=100, help="hidden size of transformer model")
    parser.add_argument("-l", "--layers", type=int, default=5, help="number of layers")
    parser.add_argument("-a", "--attn_heads", type=int, default=10, help="number of attention heads")
    parser.add_argument("-s", "--seq_len", type=int, default=1898, help="maximum sequence len")

    parser.add_argument("-b", "--batch_size", type=int, default=3, help="number of batch_size")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("-w", "--num_workers", type=int, default=0, help="dataloader worker size")

    parser.add_argument("--cuda", dest='with_cuda', action='store_true', help="train with CUDA")

    parser.add_argument("--log_freq", type=int, default=1, help="printing loss every n iter: setting n")
    parser.add_argument("--checkpoint_freq", type=int, default=5, help="checkpointing frequency every n epochs")
    parser.add_argument("--save_freq", type=int, default=10, help="saving frequency to save model after every n epochs")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")
    parser.add_argument("--log_file", type=str, default=None, help="log file for performance metrics")

    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate of adam")
    parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")

    parser.add_argument("--load_gen", type=str, default=None, help="path to saved state_dict of Masked LM model")
    parser.add_argument("--load_g_embed", type=str, default=None,
                        help="path to saved state dict for generator embeddings")
    parser.add_argument("--resume_epoch", type=int, default=0, help="epoch to resume training at (auto‑overridden by "
                                                                    "checkpoints)")
    parser.add_argument("--use_wandb", type=bool, default=False, help="use wandb logging")
    parser.add_argument("--slurm_checkpoint", type=str, default='outputs/checkpoints/',
                        help="path to slurm checkpoint for preemption")
    parser.add_argument("--wandb_savepoint", type=str, default='outputs/_wandb_runid.txt',
                        help="path to saved wandb info")

    args = parser.parse_args()


    print("Loading Train Dataset", args.train_dataset)
    train_dataset = ELECTRADataset(args.train_dataset, args.vocab_path)

    print("Loading Test Dataset", args.test_dataset)
    test_dataset = ELECTRADataset(args.test_dataset, args.vocab_path) if args.test_dataset is not None else None

    print("Creating Dataloader")
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                   drop_last=True)
    test_data_loader = DataLoader(test_dataset, batch_size=1, num_workers=args.num_workers, drop_last=True) \
        if test_dataset is not None else None

    vocab_len = train_dataset.vocab_len()

    if not os.path.exists(args.slurm_checkpoint):
        os.mkdir(args.slurm_checkpoint)

    electra_config = ElectraConfig(vocab_size=vocab_len, embedding_size=args.hidden, hidden_size=2 * args.hidden,
                                   num_hidden_layers=args.layers, num_attention_heads=args.attn_heads,
                                   intermediate_size=4 * args.hidden, max_position_embeddings=args.seq_len, )
    # electra = ElectraGenerator(electra_config, torch.from_numpy(train_dataset.embeddings), args.load_gen,
    #                            args.load_g_embed)
    electra_generator, resume_epoch = init_or_load_generator(checkpoint_folder_path=pathlib.Path(args.slurm_checkpoint),
                                                             vocab=train_dataset.embeddings,
                                                             model_config=electra_config)
    print("Creating Electra Trainer")
    if args.use_wandb:
        run = init_or_resume_wandb_run(wandb_id_file_path=pathlib.Path(args.wandb_savepoint),
                                       project_name='Electra-Generator-Microbiome-HMC',
                                       run_name='full',
                                       config=vars(args)
                                       )
    else:
        run = None

    append = True if args.resume_epoch > 0 else False
    trainer = GeneratorTrainer(electra_generator,
                               vocab_len,
                               train_dataloader=train_data_loader,
                               test_dataloader=test_data_loader,
                               lr=args.lr,
                               betas=(args.adam_beta1, args.adam_beta2),
                               weight_decay=args.adam_weight_decay,
                               with_cuda=args.with_cuda,
                               cuda_devices=args.cuda_devices,
                               log_freq=args.log_freq,
                               log_file=args.log_file,
                               append=append,
                               wandb_run=run)

    print("Training Start")
    for epoch in range(resume_epoch, args.epochs):
        trainer.train(epoch)
        if epoch == 0 or (epoch + 1) % args.checkpoint_freq == 0:
            trainer.save_slurm(epoch, args.slurm_checkpoint)
        if epoch == 0 or (epoch + 1) % args.save_freq == 0:
            trainer.save(epoch, args.output_path)
        if test_data_loader is not None:
            trainer.test(epoch)


if __name__ == "__main__":
    train()
