"""
Checkpoint management and training state persistence.
"""
import os
import shutil
from pathlib import Path
from trainers import logger
from trainers.train_functions import (
    create_or_restore_training_state_wandb,
    commit_state
)


def setup_directories(cfg, accelerator):
    """
    Create necessary directories and handle start_over flag.
    
    :param cfg: Configuration object.
    :param accelerator: Accelerator instance.
    """
    # Handle start_over flag - delete existing checkpoints
    if cfg.debug.get('start_over', False):
        logger.info("Starting over from scratch, deleting existing training state.")
        if os.path.exists(cfg.paths.data_restore_dir):
            shutil.rmtree(cfg.paths.data_restore_dir)
        if os.path.exists(cfg.paths.checkpoint_dir):
            shutil.rmtree(cfg.paths.checkpoint_dir)
    
    # Create directories
    if accelerator.is_main_process:
        Path(cfg.paths.best_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg.paths.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg.paths.intermediate_dir).mkdir(parents=True, exist_ok=True)
        logger.info("Directories created/verified")


def restore_or_initialize_state(cfg, model_config, train_loader, accelerator):
    """
    Restore from checkpoint or initialize fresh training state.
    
    :param cfg: Configuration object.
    :param model_config: Model configuration dictionary.
    :param train_loader: Training data loader (for computing total steps).
    :param accelerator: Accelerator instance.
    :return: Dictionary containing training state components.
    """
    from utils.config_utils import get_wandb_config
    
    total_steps = len(train_loader) * cfg.training.max_epochs
    
    model, optimizer, scheduler, epoch, best_val_loss, patience_counter, extra_state = \
        create_or_restore_training_state_wandb(
            model_config=model_config,
            init_lr=cfg.training.init_lr,
            warmup_ratio_or_step=cfg.training.cosine_warmup_ratio_or_step,
            total_steps=total_steps,
            checkpoint_dir=cfg.paths.checkpoint_dir,
            use_wandb=cfg.wandb.enabled,
            wandb_entity=cfg.wandb.get('entity', None),
            wandb_project=cfg.wandb.get('project', 'microbiome-pretrain'),
            wandb_config=get_wandb_config(cfg),
            accelerator=accelerator,
            wandb_run_name=cfg.wandb.get('run_name', None),
            wandb_run_notes=cfg.wandb.get('run_notes', None)
        )
    
    return {
        'model': model,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'epoch': epoch,
        'best_val_loss': best_val_loss,
        'patience_counter': patience_counter,
        'extra_state': extra_state,
        'train_loader': None,  # Will be filled by caller
        'valid_loader': None,  # Will be filled by caller
    }
