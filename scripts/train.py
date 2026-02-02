"""
Microbiome Representation Learning Training Script
Main entry point for pretraining transformer models on microbiome data.
"""
import argparse
import torch
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from accelerate import Accelerator

from utils.config_utils import load_and_validate_config, save_training_artifacts
from utils.model_utils import build_model_config, initialize_training_components
from utils.checkpoint_utils import setup_directories
from utils.data_pipeline import prepare_microbiome_data
from trainers.trainer import MicrobiomeTrainer
from trainers import logger


def setup_training_environment(cfg):
    """
    Initialize training environment including seeds, accelerator, and directories.
    
    :param cfg: OmegaConf configuration object.
    :return: Initialized Accelerator instance.
    """
    # Set seeds for reproducibility
    torch.manual_seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)
    
    # Initialize distributed training accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.grad_accumulation_steps,
        mixed_precision="fp16" if cfg.training.enable_fp16 else "no",
        log_with="wandb" if cfg.wandb.enabled else None
    )
    
    # Setup directories and handle start_over flag
    setup_directories(cfg, accelerator)
    
    return accelerator


def main(cfg):
    """
    Main training workflow for microbiome representation learning.
    
    :param cfg: OmegaConf configuration object containing all hyperparameters.
    """
    accelerator = setup_training_environment(cfg)
    
    
    logger.info("Preparing microbiome data...")
    data_artifacts = prepare_microbiome_data(
        cfg=cfg,
        accelerator=accelerator
    )
    
    train_loader = data_artifacts['train_loader']
    valid_loader = data_artifacts['valid_loader']
    taxa_vocab = data_artifacts['taxa_vocab']
    batch_vocab = data_artifacts.get('batch_vocab', None)
    graph_data = data_artifacts.get('graph_data', None)
    
    logger.info("Building model configuration...")
    model_config = build_model_config(
        cfg=cfg,
        taxa_vocab=taxa_vocab,
        batch_vocab=batch_vocab,
        graph_data=graph_data
    )
    
    if accelerator.is_main_process:
        save_training_artifacts(cfg, model_config)
    
    logger.info("Initializing training components...")
    total_steps = len(train_loader) * cfg.training.max_epochs
    training_state = initialize_training_components(
        model_config=model_config,
        cfg=cfg,
        total_steps=total_steps,
        accelerator=accelerator
    )
    
    prepared_train_loader, prepared_valid_loader, prepared_model, prepared_optimizer, prepared_scheduler = accelerator.prepare(
        train_loader,
        valid_loader,
        training_state['model'],
        training_state['optimizer'],
        training_state['scheduler']
    )
    
    if graph_data is not None:
        graph_data = graph_data.to(accelerator.device)
    
    trainer = MicrobiomeTrainer(
        cfg=cfg,
        accelerator=accelerator,
        taxa_vocab=taxa_vocab,
        batch_vocab=batch_vocab,
        graph_data=graph_data
    )
    
    logger.info("Starting training...")

    trainer.train(
        model=prepared_model,
        train_loader=prepared_train_loader,
        valid_loader=prepared_valid_loader,
        optimizer=prepared_optimizer,
        scheduler=prepared_scheduler,
        start_epoch=training_state['epoch'],
        best_val_loss=training_state['best_val_loss'],
        patience_counter=training_state['patience_counter']
    )
    
    logger.info("Training complete!")
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train microbiome transformer model with flexible YAML config"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Override config values (e.g., training.batch_size=64 model.tasks.do_mvc=True)"
    )
    
    args = parser.parse_args()

    # Load and merge configurations
    cfg = load_and_validate_config(args.config, args.overrides)
    
    # Run training
    main(cfg)