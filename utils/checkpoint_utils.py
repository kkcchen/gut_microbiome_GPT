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
    # given cfg.paths.output_dir, set up subdirectories
    cfg.paths.best_dir = os.path.join(cfg.paths.output_dir, "best_model")
    cfg.paths.checkpoint_dir = os.path.join(cfg.paths.output_dir, "checkpoints")
    cfg.paths.intermediate_dir = os.path.join(cfg.paths.output_dir, "intermediate")
    cfg.paths.data_restore_dir = os.path.join(cfg.paths.output_dir, "data_cache")
    cfg.paths.taxa_vocab_path = os.path.join(cfg.paths.output_dir, "taxa_vocab.pkl")
    cfg.paths.batch_vocab_path = os.path.join(cfg.paths.output_dir, "batch_vocab.pkl")
    cfg.paths.graph_path = os.path.join(cfg.paths.output_dir, "taxonomic_graph.pt")

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
    
    return cfg


def setup_directories_eval(cfg, accelerator):
    """
    Create necessary directories for inference and handle start_over flag.
    
    :param cfg: Configuration object.
    :param accelerator: Accelerator instance.
    """
    # given cfg.paths.output_dir, set up subdirectories
    cfg.paths.taxa_vocab_path = os.path.join(cfg.paths.pretrained_model_dir, "taxa_vocab.pkl")
    cfg.paths.batch_vocab_path = os.path.join(cfg.paths.pretrained_model_dir, "batch_vocab.pkl")
    cfg.paths.model_config_path = os.path.join(cfg.paths.output_dir, "model_config.json")

    if cfg.eval.get('overwrite_output_dir', False):
        logger.info("Overwrite output directory is enabled, deleting existing inference outputs.")
        if os.path.exists(cfg.paths.output_dir):
            shutil.rmtree(cfg.paths.output_dir)
    
    # Create directories
    if accelerator.is_main_process:
        Path(cfg.paths.output_dir).mkdir(parents=True, exist_ok=True)
        logger.info("Eval output directory created/verified")

    # also check if pretrained_model_dir exists (for loading vocabularies)
    if not os.path.exists(cfg.paths.pretrained_model_dir):
        raise FileNotFoundError(f"Pretrained model directory {cfg.paths.pretrained_model_dir} does not exist. Cannot load vocabularies.")
    
    # also check if checkpoint path exists
    if not os.path.exists(cfg.paths.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint path {cfg.paths.checkpoint_path} does not exist. Cannot load model.")    
    return cfg

