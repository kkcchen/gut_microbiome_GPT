import argparse
import pathlib
from typing import Optional, Dict, List
from torch.utils.data import DataLoader

import torch
from trainers import DiscriminatorTrainer
from data_utils import ELECTRADataset
from transformers import ElectraConfig
from models import ElectraDiscriminator
from models import ElectraGenerator
import wandb
import os
import numpy as np


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


def init_or_load_discriminator(checkpoint_folder_path: pathlib.Path,
                               vocab: np.array,
                               generator_ckpt_paths: List[pathlib.Path],
                               generator_embed_paths: List[pathlib.Path],
                               generator_config: ElectraConfig,
                               discriminator_config: ElectraConfig):
    """Detect the previous checkpointing for slurm
        checkpoint_folder should contain
        - latest discriminator model
        - latest discriminator embed
        - latest generator model
        - latest generator embed
        also resume from the appropriate epoch count
        - appropriate generator epochs
        - appropriate discriminator epochs
        Returns DiscriminatorTrainer, with appropripate resume of training
    """
    # Paths inside the folder
    disc_embed_path = checkpoint_folder_path / "disc_embed"
    disc_ckpt_path = checkpoint_folder_path / "disc_model"
    resume_path = checkpoint_folder_path / "resume_from.txt"

    can_resume = (
            disc_embed_path.is_file() and
            disc_ckpt_path.is_dir() and
            resume_path.is_file()
    )

    if can_resume:
        # Load the epoch to resume from
        gen_idx_str, disc_epoch_str = resume_path.read_text().strip().split(",")
        gen_idx = int(gen_idx_str)
        disc_epoch = int(disc_epoch_str)
        if not (0 <= gen_idx < len(generator_ckpt_paths)):
            raise ValueError(f"resume gen_idx {gen_idx} is out of bounds for provided ckpt list")

        # Load embeddings and model
        gen_ckpt_path = generator_ckpt_paths[gen_idx]
        gen_embed_path = generator_embed_paths[gen_idx]
        if not (gen_embed_path.is_file() and gen_ckpt_path.is_dir()):
            raise FileNotFoundError(f"Generator checkpoint not found at {gen_ckpt_path}")
        generator_model = ElectraGenerator(generator_config,
                                           torch.from_numpy(vocab),
                                           gen_ckpt_path,
                                           gen_embed_path)
        discriminator_model = ElectraDiscriminator(discriminator_config, torch.from_numpy(vocab), disc_ckpt_path,
                                                   disc_embed_path)
        print(f"resuming training discriminator for generator {gen_ckpt_path} at epoch {disc_epoch}")

    else:
        # Fresh start
        gen_idx = 0
        disc_epoch = 0
        gen_ckpt_path = generator_ckpt_paths[gen_idx]
        gen_embed_path = generator_embed_paths[gen_idx]
        if not (gen_embed_path.is_file() and gen_ckpt_path.is_dir()):
            raise FileNotFoundError(f"Generator checkpoint not found at {gen_ckpt_path}")
        generator_model = ElectraGenerator(generator_config, torch.from_numpy(vocab), generator=gen_ckpt_path,
                                           embed_layer=gen_embed_path)
        discriminator_model = ElectraDiscriminator(discriminator_config, torch.from_numpy(vocab))
        print("No checkpoint found, starting training from beginning")

    return generator_model, discriminator_model, gen_idx, disc_epoch


def train():
    """
    Pretrain an ELECTRA discriminator model.

    This function sets up and executes the training process for an ELECTRA discriminator model, using a sequence of generator models provided by the user. It handles argument parsing, dataset loading, model initialization, and the training loop.

    The function performs the following steps:
    1. Parse command-line arguments for training configuration.
    2. Load and prepare the training and test datasets.
    3. Create data loaders for efficient batch processing.
    4. Initialize the ELECTRA generator and discriminator models.
    5. Set up the ELECTRA trainer with the specified parameters.
    6. Execute the training loop, which involves:
       - Loading different generator models (if specified)
       - Training the discriminator for a set number of epochs
       - Saving the model checkpoints

    The training process can be customized through various command-line arguments,
    including model architecture, training hyperparameters, and data paths.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--train_dataset", required=True, type=str, help="train dataset for train electra")
    parser.add_argument("-t", "--test_dataset", type=str, default=None, help="test set for evaluate train set")
    parser.add_argument("-v", "--vocab_path", required=True, type=str, help="built vocab model path with electra-vocab")
    parser.add_argument("-o", "--output_path", required=True, type=str, help="ex)output/")

    parser.add_argument("-hs", "--hidden", type=int, default=100, help="hidden size of transformer model")
    parser.add_argument("-l", "--layers", type=int, default=5, help="number of layers")
    parser.add_argument("-a", "--attn_heads", type=int, default=10, help="number of attention heads")
    parser.add_argument("-s", "--seq_len", type=int, default=1898, help="maximum sequence len")

    parser.add_argument("-ghs", "--gen_hidden", type=int, default=100, help="hidden size of gen transformer model")
    parser.add_argument("-gl", "--gen_layers", type=int, default=10, help="number of gen layers")
    parser.add_argument("-ga", "--gen_attn_heads", type=int, default=10, help="number of gen attention heads")
    parser.add_argument("-gs", "--gen_seq_len", type=int, default=1898, help="maximum gen sequence len")

    parser.add_argument("-b", "--batch_size", type=int, default=3, help="number of batch_size")
    parser.add_argument("-e", "--epochs", type=int, default=10,
                        help="number of epochs to train discriminator for each generator")
    parser.add_argument("-w", "--num_workers", type=int, default=0, help="dataloader worker size")
    parser.add_argument("--freeze", dest='freeze_embed', action='store_true',
                        help="freeze discriminator embedding layer to GloVE embeddings")
    parser.add_argument("--no_freeze", dest='freeze_embed', action='store_false',
                        help="train discriminator embedding layer after initializing to GloVE embeddings")
    parser.set_defaults(freeze_embed=False)

    parser.add_argument("--cuda", dest='with_cuda', action='store_true', help="train with CUDA")
    parser.add_argument("--no_cuda", dest='with_cuda', action='store_false', help="train on CPU")
    parser.set_defaults(with_cuda=False)

    parser.add_argument("--log_freq", type=int, default=100, help="printing loss every n iter: setting n")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")
    parser.add_argument("--d_log_file", type=str, default=None, help="log file for discriminator performance metrics")

    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate of adam")
    parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")

    parser.add_argument("--load_gen", type=str, nargs='+', default=None,
                        help="path to saved state_dicts of Masked LM model")
    parser.add_argument("--load_g_embed", type=str, nargs='+', default=None,
                        help="path to saved state_dicts for generator embeddings")
    parser.add_argument("--resume_epoch", type=int, default=0, help="epoch to resume training at")

    parser.add_argument("--use_wandb", type=bool, default=False, help="use wandb logging")
    parser.add_argument("--wandb_project", type=str, default=None, help="wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="wandb run name")
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

    discriminator_config = ElectraConfig(vocab_size=vocab_len, embedding_size=args.hidden, hidden_size=2 * args.hidden,
                                         num_hidden_layers=args.layers, num_attention_heads=args.attn_heads,
                                         intermediate_size=4 * args.hidden, max_position_embeddings=args.seq_len)
    generator_config = ElectraConfig(vocab_size=vocab_len, embedding_size=args.gen_hidden,
                                     hidden_size=2 * args.gen_hidden, num_hidden_layers=args.gen_layers,
                                     num_attention_heads=args.gen_attn_heads, intermediate_size=4 * args.gen_hidden,
                                     max_position_embeddings=args.gen_seq_len)
    generator_ckpt_paths = [pathlib.Path(p) for p in args.load_gen]
    generator_embed_paths = [pathlib.Path(p) for p in args.load_g_embed]
    electra_gen, electra_disc, resume_gen_id, resume_epoch = init_or_load_discriminator(checkpoint_folder_path=pathlib.Path(args.wandb_savepoint),
                                                                                        vocab=train_dataset.embeddings,
                                                                                        generator_ckpt_paths=generator_ckpt_paths,
                                                                                        generator_embed_paths=generator_embed_paths,
                                                                                        generator_config=generator_config,
                                                                                        discriminator_config=discriminator_config)
    # electra_gen = ElectraGenerator(electra_gen_config,
    #                                torch.from_numpy(train_dataset.embeddings),
    #                                args.load_gen[0],
    #                                args.load_g_embed[0])
    # electra_disc = ElectraDiscriminator(electra_config, torch.from_numpy(train_dataset.embeddings))

    print("Creating Electra Trainer")
    if args.use_wandb:
        wandb_project_name = 'Electra-Generator-Microbiome-HMC'
        wandb_run_name = 'discriminator-full'
        if args.wandb_project is not None:
            wandb_project_name = args.wandb_project
        if args.wandb_run_name is not None:
            wandb_run_name = args.wandb_run_name
        run = init_or_resume_wandb_run(wandb_id_file_path=pathlib.Path(args.wandb_savepoint),
                                       project_name=wandb_project_name,
                                       run_name=wandb_run_name,
                                       config=vars(args)
                                       )
    else:
        run = None

    append = True if args.resume_epoch > 0 else False
    trainer = DiscriminatorTrainer(electra_gen, electra_disc, vocab_len, train_dataloader=train_data_loader,
                                   test_dataloader=test_data_loader,
                                   lr=args.lr, betas=(args.adam_beta1, args.adam_beta2),
                                   weight_decay=args.adam_weight_decay,
                                   with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=args.log_freq,
                                   d_log_file=args.d_log_file,
                                   append=append, freeze_embed=args.freeze_embed, wandb_run=run)

    for i in range(len(args.load_gen)):
        new_gen = ElectraGenerator(generator_config, torch.from_numpy(train_dataset.embeddings), args.load_gen[i],
                                   args.load_g_embed[i])
        trainer.swap_gen(new_gen, args.with_cuda, args.cuda_devices)
        trainer.save(0, args.output_path)
        for j in range(args.epochs):
            epoch_num = i * args.epochs + (j + 1)
            # print(epoch_num)
            trainer.train(epoch_num)
            trainer.test(epoch_num)
        trainer.save((i + 1) * args.epochs, args.output_path)


if __name__ == "__main__":
    train()
