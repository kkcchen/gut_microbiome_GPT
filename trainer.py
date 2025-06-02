from torch.utils.data import DataLoader
import torch

import os
import shutil
import numpy as np
import time

from data_utils.preprocessor import Preprocessor
from data_utils.dataloader import prepare_dataloader

from sklearn.model_selection import train_test_split

from trainers.train_functions import (
    pretrain, commit_state, create_or_restore_data_state_and_wandb, create_or_restore_training_state, epoch_end_logs
)
from trainers import logger

from accelerate import Accelerator

import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train TransformerModel on microbiome data")
    parser.add_argument("--hmc-table-path", type=str, required=True, help="Path to HMC table file")
    parser.add_argument("--taxa-path", type=str, required=True, help="Path to taxa file")
    parser.add_argument("--best-dir", type=str, required=True, help="Directory to save best model so far")
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="Directory to save checkpoints for preemption")
    parser.add_argument("--data-restore-path", type=str, required=True, help="Path to restore data state")

    # wandb
    parser.add_argument("--wandb-enabled", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-entity", type=str, default=None, help="wandb entity name")
    parser.add_argument("--wandb-project", type=str, default=None, help="wandb project name")

    # optional
    parser.add_argument("--init-lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--max-epochs", type=int, default=25, help="Maximum number of epochs")
    parser.add_argument("--cosine-warmup-ratio-or-step", type=float, default=0.1, help="Scheduler warmup ratio or step")
    parser.add_argument("--num-bins", type=int, default=10, help="Number of bins for binning")
    parser.add_argument("--log-interval", type=int, default=10, help="Interval for logging")
    parser.add_argument("--patience", type=int, default=None, help="Patience for early stopping")
    parser.add_argument("--grad-accumulation-steps", type=int, default=1, help="Number of gradient accumulation steps")
    parser.add_argument("--enable-fp16", action="store_true", help="Enable mixed precision training (FP16)")

    # for debugging
    parser.add_argument("--nrows", type=int, default=None, help="For debugging to limit number of samples in set")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--start-over", action="store_true", help="Start over from scratch, ignoring existing data and checkpoints")

    args = parser.parse_args()

    hmc_table_path = args.hmc_table_path
    taxa_path = args.taxa_path
    best_dir = args.best_dir
    checkpoint_dir = args.checkpoint_dir
    data_restore_path = args.data_restore_path

    # Create directories if they don't exist
    if data_restore_path is not None:
        os.makedirs(os.path.dirname(data_restore_path), exist_ok=True)
    # accelerator will create the checkpoint directory if it doesn't exist

    wandb_enabled = args.wandb_enabled
    wandb_entity = args.wandb_entity
    wandb_project = args.wandb_project

    init_lr = args.init_lr
    batch_size = args.batch_size
    max_epochs = args.max_epochs
    cosine_warmup_ratio_or_step = args.cosine_warmup_ratio_or_step
    num_bins = args.num_bins
    log_interval = args.log_interval
    patience = args.patience if args.patience else max_epochs
    grad_accumulation_steps = args.grad_accumulation_steps
    enable_fp16 = args.enable_fp16

    nrows = args.nrows
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.start_over:
        logger.info("Starting over from scratch, deleting existing training state.")
        if os.path.exists(data_restore_path):
            os.remove(data_restore_path)
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)

    accelerator = Accelerator(gradient_accumulation_steps=grad_accumulation_steps, mixed_precision="fp16" if enable_fp16 else "no", log_with="wandb" if wandb_enabled else None)

    # Create or restore data state and wandb
    train_data_dict, valid_data_dict, vocab, run = create_or_restore_data_state_and_wandb(
        hmc_table_path, taxa_path, wandb_enabled, wandb_entity, wandb_project, init_lr, batch_size, max_epochs, cosine_warmup_ratio_or_step, num_bins, data_restore_path, accelerator, nrows
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

    # Create or restore training state
    model, optimizer, scheduler, epoch, best_val_loss, patience_counter = create_or_restore_training_state(
        vocab, init_lr, cosine_warmup_ratio_or_step, max_epochs, len(train_loader), checkpoint_dir, accelerator
    )

    train_loader, valid_loader, model, optimizer, scheduler = accelerator.prepare(
        train_loader, valid_loader, model, optimizer, scheduler
    )

    while epoch < max_epochs:
        logger.info(f"Epoch {epoch + 1}/{max_epochs}")
        epoch_start_time = time.time()

        # Train the model
        val_loss, val_mre = pretrain(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            epoch=epoch,
            log_interval=log_interval,
            vocab=vocab,
            accelerator=accelerator,
            optimizer=optimizer,
            scheduler=scheduler,
            best_dir=best_dir,
            logger=logger,
            epoch_start_time=epoch_start_time,
            best_val_loss=best_val_loss,
        )

        # Log metrics to wandb
        epoch_end_logs(epoch_start_time, epoch, val_loss, val_mre)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            logger.info(f"New best validation loss: {best_val_loss:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            logger.info("Early stopping triggered. Stopping training.")
            break

        epoch += 1
        commit_state(model, optimizer, scheduler, epoch, best_val_loss, patience_counter, checkpoint_dir, accelerator)

    logger.info("Training complete with best validation loss: {:.4f}".format(best_val_loss))
    accelerator.end_training()