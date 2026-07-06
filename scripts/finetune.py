"""
Microbiome Representation Learning Finetuning Script

Finetune pretrained transformer models on supervised downstream tasks.
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from accelerate import Accelerator

from utils.config_utils import load_and_validate_config, save_training_artifacts
from utils.model_utils import build_model_config
from utils.checkpoint_utils import setup_directories

from trainers.trainer import MicrobiomeTrainer
from trainers import logger
from utils.data_pipeline import prepare_finetune_data
from utils.model_utils import load_pretrained_model_for_finetune, initialize_finetuning_components

def setup_finetuning_environment(cfg):
    """
    Initialize finetuning environment including seeds, accelerator, and directories.
    
    :param cfg: OmegaConf configuration object.
    :return: Updated config and initialized Accelerator instance.
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
    
    # Setup directories
    cfg = setup_directories(cfg, accelerator,finetune=True)
    
    return cfg, accelerator


def main(cfg):
    """
    Main finetuning workflow.
    
    :param cfg: OmegaConf configuration object containing all hyperparameters.
    """
    cfg, accelerator = setup_finetuning_environment(cfg)
    
    logger.info("=" * 80)
    logger.info("MICROBIOME FINETUNING - SUPERVISED LEARNING")
    logger.info("=" * 80)
    logger.info(f"Task type: {cfg.training.finetune_task}")
    logger.info(f"Finetune mode: {cfg.training.finetune_mode}")
    if "checkpoint_path" in cfg.paths and cfg.paths.checkpoint_path is not None:
        logger.info(f"Pretrained checkpoint: {cfg.paths.checkpoint_path}")
    
    # Prepare data
    logger.info("\nPreparing finetuning data...")
    data_artifacts = prepare_finetune_data(cfg, accelerator)
    
    train_loader = data_artifacts['train_loader']
    valid_loader = data_artifacts['valid_loader']
    test_loader = data_artifacts.get('test_loader', None)  # Optional
    taxa_vocab = data_artifacts['taxa_vocab']
    batch_vocab = data_artifacts['batch_vocab']
    graph_data = data_artifacts['graph_data']
    num_classes = data_artifacts['num_classes']
    
    # Build model configuration
    logger.info("\nBuilding model configuration...")
    model_config = build_model_config(
        cfg=cfg,
        taxa_vocab=taxa_vocab,
        batch_vocab=batch_vocab,
        graph_data=graph_data
    )
    
    # Add finetuning-specific parameters
    model_config['finetune_mode'] = cfg.training.finetune_mode
    model_config['finetune_task'] = cfg.training.finetune_task
    if cfg.training.finetune_task == 'classification':
        model_config['finetune_num_classes'] = num_classes
    
    # Load pretrained model
    model = load_pretrained_model_for_finetune(cfg, model_config, accelerator)
    
    # Save configuration
    if accelerator.is_main_process:
        save_training_artifacts(cfg, model_config)
    
    # Initialize optimizer and scheduler
    logger.info("\nInitializing training components...")
    total_steps = len(train_loader) * cfg.training.max_epochs
    training_components = initialize_finetuning_components(model, cfg, total_steps)
    
    optimizer = training_components['optimizer']
    scheduler = training_components['scheduler']
    
    # Prepare with accelerator
    prepared_train_loader, prepared_valid_loader, prepared_model, prepared_optimizer, prepared_scheduler = accelerator.prepare(
        train_loader,
        valid_loader,
        model,
        optimizer,
        scheduler
    )
    # Prepare test loader if exists
    if test_loader is not None:
        prepared_test_loader = accelerator.prepare(test_loader)
    else:
        prepared_test_loader = None
    
    if graph_data is not None:
        graph_data = graph_data.to(accelerator.device)
    
    # Create trainer (reuse existing MicrobiomeTrainer)
    trainer = MicrobiomeTrainer(
        cfg=cfg,
        accelerator=accelerator,
        taxa_vocab=taxa_vocab,
        batch_vocab=batch_vocab,
        graph_data=graph_data,
    )
    
    # Start finetuning
    logger.info("\n" + "=" * 80)
    logger.info("STARTING FINETUNING")
    logger.info("=" * 80)
    
    trainer.train(
        model=prepared_model,
        train_loader=prepared_train_loader,
        valid_loader=prepared_valid_loader,
        optimizer=prepared_optimizer,
        scheduler=prepared_scheduler,
        start_epoch=0,
        best_val_loss=float('inf'),
        patience_counter=0
    )
    
    logger.info("\n" + "=" * 80)
    logger.info("FINETUNING COMPLETE!")
    logger.info("=" * 80)

    # Evaluate on test set if available
    if test_loader is not None and prepared_test_loader is not None:
        logger.info("Evaluating on test set...")
        test_metrics = trainer.evaluate_on_test_set(
            model=prepared_model,
            test_loader=prepared_test_loader,
            load_best_checkpoint=True,
            best_model_path=None  # Will use default from trainer config
        )
        
        # Log to wandb if enabled
        if cfg.wandb.enabled and accelerator.is_main_process:
            accelerator.log(test_metrics, step=trainer.global_step)
    else:
        logger.warning("No test loader provided, skipping test evaluation")

    
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Finetune pretrained microbiome transformer on supervised tasks"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file for finetuning"
    )
    
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Override config values (e.g., finetuning.learning_rate=1e-4)"
    )
    
    args = parser.parse_args()
    
    # Load and merge configurations
    cfg = load_and_validate_config(args.config, args.overrides)
    
    # Run finetuning
    main(cfg)
