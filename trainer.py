from torch.utils.data import DataLoader
import torch

import os
import numpy as np
import json

from data_utils.preprocessor import Preprocessor
from data_utils.dataloader import prepare_dataloader

from sklearn.model_selection import train_test_split

from trainers.train_functions import (
    pretrain, commit_state, create_or_restore_data_state_and_wandb, create_or_restore_training_state
)
from trainers import logger

from models import TransformerModel
from data_utils.tokenizer import MicrobiomeVocab, Tokenizer

import wandb
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train TransformerModel on microbiome data")
    parser.add_argument("--hmc_table_path", type=str, required=True, help="Path to HMC table file")
    parser.add_argument("--taxa_path", type=str, required=True, help="Path to taxa file")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save models")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Directory to save checkpoints for preemption")
    parser.add_argument("--data_restore_path", type=str, default=None, help="Path to restore data state")

    parser.add_argument("--wandb_enabled", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_entity", type=str, default=None, help="wandb entity name")
    parser.add_argument("--wandb_project", type=str, default=None, help="wandb project name")

    parser.add_argument("--init_lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--max_epochs", type=int, default=25, help="Maximum number of epochs")
    parser.add_argument("--scheduler_interval", type=int, default=25, help="Scheduler step interval")
    parser.add_argument("--num_bins", type=int, default=10, help="Number of bins for binning")
    parser.add_argument("--log_interval", type=int, default=10, help="Interval for logging")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")

    parser.add_argument("--nrows", type=int, default=None, help="For debugging to limit number of samples in set")

    args = parser.parse_args()

    hmc_table_path = args.hmc_table_path
    taxa_path = args.taxa_path
    save_dir = args.save_dir
    checkpoint_path = args.checkpoint_path
    data_restore_path = args.data_restore_path

    # Create directories if they don't exist
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    if data_restore_path is not None:
        os.makedirs(os.path.dirname(data_restore_path), exist_ok=True)

    wandb_enabled = args.wandb_enabled
    wandb_entity = args.wandb_entity
    wandb_project = args.wandb_project

    init_lr = args.init_lr
    batch_size = args.batch_size
    max_epochs = args.max_epochs
    scheduler_interval = args.scheduler_interval
    num_bins = args.num_bins
    log_interval = args.log_interval
    patience = args.patience

    nrows = args.nrows

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create or restore data state and wandb
    train_data_dict, valid_data_dict, vocab, run = create_or_restore_data_state_and_wandb(
        hmc_table_path, taxa_path, wandb_enabled, wandb_entity, wandb_project, init_lr, batch_size, max_epochs, scheduler_interval, num_bins, data_restore_path, nrows
    )

    # Create or restore training state
    model, optimizer, scaler, scheduler, epoch, best_val_loss, patience_counter = create_or_restore_training_state(
        vocab, init_lr, scheduler_interval, device, checkpoint_path
    )

    logger.info("Preparing dataloaders...")
    train_loader = prepare_dataloader(
        train_data_dict,
        vocab=vocab,
        batch_size=batch_size,
        shuffle=True,
    )
    valid_loader = prepare_dataloader(
        valid_data_dict,
        vocab=vocab,
        batch_size=batch_size,
        shuffle=False,
    )

    while epoch < max_epochs:
        logger.info(f"Epoch {epoch + 1}/{max_epochs}")
        logger.info("Training...")

        rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state() if device == "cuda" else None

        # Train the model
        val_loss, val_mre = pretrain(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            epoch=epoch,
            log_interval=log_interval,
            vocab=vocab,
            enable_fp16=False,
            grad_accu_steps=1,
            scaler=scaler,
            optimizer=optimizer,
            scheduler=scheduler,
            save_dir=save_dir,
            device=device,
            logger=logger,
            best_val_loss=best_val_loss,
        )

        # Log metrics to wandb
        wandb.log({
            "val_loss": val_loss,
            "val_mre": val_mre,
            "learning_rate": scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else init_lr,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            logger.info(f"New best validation loss: {best_val_loss:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            logger.info("Early stopping triggered. Stopping training.")
            break

        commit_state(model, optimizer, scheduler, scaler, rng, cuda_rng, epoch, best_val_loss, patience_counter, checkpoint_path)

    logger.info("Training complete with best validation loss: {:.4f}".format(best_val_loss))
    wandb.finish()