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


if __name__ == "__main__":
    # Example usage
    # wandb.init(project="hmbGPT", config={
    #     "learning_rate": 0.001,
    #     "batch_size": 32,
    #     "epochs": 10,
    #     "scheduler_interval": 100
    # })

    data_dir = "/home/kchen/microbiome/gut_microbiome_GPT/datasets"
    hmc_table_path = os.path.join(data_dir, "taxonomy_table_512.npy")
    hmc_npy = np.load(hmc_table_path)

    taxa_path = os.path.join(data_dir, "taxonomy_table_512_taxa.json")
    with open(taxa_path, "r") as f:
        taxa_list = json.load(f)


    # config = wandb.config
    config = {
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 1,
        "scheduler_interval": 100
    }

    learning_rate = config["learning_rate"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    save_dir = "./model_checkpoints"

    log_interval = 1
    save_interval = 1

    preprocessor = Preprocessor(
        binning=10,
    )

    binned_data, bin_edges = preprocessor.process_from_np(hmc_npy)

    vocab = MicrobiomeVocab(taxa_list)  # Replace with your vocab
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Using device: {device}")

    # create tokenizer
    tokenizer = Tokenizer(vocab)
    data_dict = tokenizer.tokenize_and_pad_batch()
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
    logger.info("Starting training for one epoch")

    for epoch in range(epochs):
        logger.info(f"Epoch {epoch + 1}/{epochs}")
        logger.info("Training...")

        # Train the model
        best_val_loss = pretrain(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            epoch=0,
            log_interval=log_interval,
            vocab=vocab,
            enable_fp16=False,
            grad_accu_steps=1,
            scaler=scaler,
            optimizer=optimizer,
            scheduler=scheduler,
            save_interval=save_interval,
            save_dir=save_dir,
            device=device,
            logger=logger,
            best_val_loss=best_val_loss,
        )

    logger.info("Training complete with best validation loss: {:.4f}".format(best_val_loss))