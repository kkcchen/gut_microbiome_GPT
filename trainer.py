from torch.utils.data import DataLoader
import torch

import os
import numpy as np
import json

from data_utils.preprocessor import Preprocessor
from data_utils.dataloader import prepare_dataloader

from sklearn.model_selection import train_test_split

from trainers.train_functions import (
    pretrain
)
from trainers import logger

from models import TransformerModel
from data_utils.tokenizer import MicrobiomeVocab, Tokenizer

import wandb

if __name__ == "__main__":
    # Example usage
    wandb.login()
    run = wandb.init(
        mode="disabled", # comment this out to enable wandb logging
        entity="kevinkaiwen-chen-vector",
        project="hmbGPT",
        config={
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 25,
            "scheduler_interval": 100
        }
    )

    data_dir = "/home/kchen/microbiome/gut_microbiome_GPT/datasets"
    hmc_table_path = os.path.join(data_dir, "taxonomy_table_512.npy")
    hmc_npy = np.load(hmc_table_path)[:64,:,:] # shape (num_samples, num_taxa, 2) where (:,:,0) is taxa_id and (:,:,1) is counts

    # taxa_path = os.path.join(data_dir, "taxonomy_table_512_taxa.json")
    # with open(taxa_path, "r") as f:
    #     taxa_list = json.load(f)

    # we don't have a taxa list for now. just use enumerate max for taxa list
    max_taxa_index = int(np.max(hmc_npy[:,:,0]))
    taxa_list = [str(i) for i in range(max_taxa_index + 1)]

    config = wandb.config
    # config = {
    #     "learning_rate": 0.001,
    #     "batch_size": 32,
    #     "epochs": 1,
    #     "scheduler_interval": 100
    # }

    learning_rate = config["learning_rate"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    save_dir = os.path.join("./model_checkpoints", f"run_{wandb.run.id}")
    os.makedirs(save_dir, exist_ok=True)
    patience = 5

    log_interval = 1

    preprocessor = Preprocessor(
        binning=10,
    )

    _, _ = preprocessor.process_from_np(hmc_npy)

    vocab = MicrobiomeVocab(taxa_list)  # Replace with your vocab
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Using device: {device}")

    # create tokenizer
    tokenizer = Tokenizer(vocab)
    data_dict = tokenizer.tokenize_and_pad_batch(hmc_npy)
    # Assuming data_dict is a dictionary with keys 'taxa_ids' and 'values'

    # train and validation split
    (
        train_taxa_ids,
        valid_taxa_ids,
        train_values,
        valid_values
    ) = train_test_split(
        data_dict["taxa_ids"],
        data_dict["values"],
        test_size=0.2,
        shuffle=True
    )

    train_data_dict = {
        "taxa_ids": train_taxa_ids,
        "values": train_values
    }
    valid_data_dict = {
        "taxa_ids": valid_taxa_ids,
        "values": valid_values
    }

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

    model = TransformerModel(
        d_model=512,
        nhead=8,
        d_hid=2048,
        nlayers=6,
        vocab=vocab,
        dropout=0.1,
        use_generative_training=True,
    )

    model.to(device)

    logger.info("model")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # can add warmup after
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["scheduler_interval"], gamma=0.1)
    scaler = torch.amp.GradScaler(device)

    best_val_loss = float("inf")
    patience_counter = 0
    logger.info("Starting training for one epoch")

    for epoch in range(epochs):
        logger.info(f"Epoch {epoch + 1}/{epochs}")
        logger.info("Training...")

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
            "learning_rate": scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else learning_rate,
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

    logger.info("Training complete with best validation loss: {:.4f}".format(best_val_loss))
    wandb.finish()