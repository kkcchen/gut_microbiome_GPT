"""
Microbiome Representation Learning Training Script
Main entry point for pretraining transformer models on microbiome data.
"""
import argparse
import sys
import torch
import numpy as np
import random
from pathlib import Path
from omegaconf import OmegaConf
from accelerate import Accelerator

from utils.config_utils import load_and_validate_config, save_training_artifacts
from utils.model_utils import build_model_config, initialize_training_components
from utils.checkpoint_utils import setup_directories
from utils.data_pipeline import prepare_microbiome_data
from utils.finetune_orchestration import require_finetune_section, require_single_process, run_all_downstream_finetunes
from trainers.trainer import MicrobiomeTrainer
from trainers import logger


def setup_training_environment(cfg):
    """
    Initialize training environment including seeds, accelerator, and directories.
    
    :param cfg: OmegaConf configuration object.
    :return: Initialized Accelerator instance.
    """
    # Set seeds for reproducibility
    seed = cfg.training.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Initialize distributed training accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.grad_accumulation_steps,
        mixed_precision="fp16" if cfg.training.enable_fp16 else "no",
        log_with="wandb" if cfg.wandb.enabled else None
    )
    
    # Setup directories and handle start_over flag
    cfg = setup_directories(cfg, accelerator)
    
    return cfg, accelerator


def main(cfg):
    """
    Main training workflow for microbiome representation learning.
    
    :param cfg: OmegaConf configuration object containing all hyperparameters.
    """
    # Fail fast: pretraining now always chains into finetuning across every task in
    # configs/finetune/task_registry.yaml, so a missing 'finetune:' block should stop
    # the run before any expensive pretraining work starts, not after.
    require_finetune_section(cfg)

    cfg, accelerator = setup_training_environment(cfg)

    # Fail fast on every rank: the auto-finetune chain deadlocks under multi-GPU DDP
    # (see utils/finetune_orchestration.require_single_process for why), so refuse to
    # even start pretraining rather than fail confusingly after hours of training.
    require_single_process(accelerator)

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
    
    # total_steps should account for gradient accumulation and multi-GPU splitting
    # BUT most importantly, initialize_training_components is called BEFORE prepare.
    # We should prepare the loader first to get the true length per process if using distributed.
    prepared_train_loader = accelerator.prepare(train_loader)
    total_steps = (len(prepared_train_loader) * cfg.training.max_epochs) // cfg.training.grad_accumulation_steps
    
    training_state = initialize_training_components(
        model_config=model_config,
        cfg=cfg,
        total_steps=total_steps,
        accelerator=accelerator
    )
    
    # Handle actual state loading if a checkpoint is provided
    if cfg.get('checkpoint_path') and Path(cfg.checkpoint_path).exists():
        accelerator.load_state(cfg.checkpoint_path)
        # Assuming our Wrapper matches the saved state keys
        training_state['epoch'] = training_state['extra_state'].data.get('epoch', 0)
        training_state['best_val_loss'] = training_state['extra_state'].data.get('best_val_loss', float('inf'))
        training_state['patience_counter'] = training_state['extra_state'].data.get('patience_counter', 0)

    # Prepare remaining components
    prepared_valid_loader, prepared_model, prepared_optimizer, prepared_scheduler = accelerator.prepare(
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
        graph_data=graph_data,
        # wandb_run=training_state['extra_state'].data['wandb_run']
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

    logger.info("Starting automatic finetuning across all downstream tasks...")
    # No-ops (returns all "pending") on non-main processes -- safe to call on every rank.
    task_results = run_all_downstream_finetunes(cfg, accelerator)
    accelerator.wait_for_everyone()

    accelerator.end_training()

    # Exit non-zero if any finetune task didn't complete, so SLURM/job monitoring reflects
    # partial failure instead of always reporting COMPLETED. Only the main process has real
    # results (see run_all_downstream_finetunes); other ranks would just see "pending" and
    # must not spuriously exit non-zero on that basis.
    if accelerator.is_main_process:
        num_failed = sum(1 for r in task_results.values() if r["status"] != "ok")
        if num_failed:
            logger.error(
                f"{num_failed}/{len(task_results)} auto-finetune tasks did not complete successfully "
                f"-- see finetune_summary.md and the logs above for details."
            )
            sys.exit(1)


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