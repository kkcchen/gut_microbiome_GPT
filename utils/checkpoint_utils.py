"""
Checkpoint management and training state persistence.
"""
import os
import shutil
from pathlib import Path
from trainers import logger


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

