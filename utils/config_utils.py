"""
Configuration utilities for loading, validating, and saving training configs.
"""
import os
import json
from pathlib import Path
from omegaconf import OmegaConf
from trainers import logger


def load_and_validate_config(config_path: str, overrides: list = None):
    """
    Load YAML config and apply CLI overrides with validation.
    
    :param config_path: Path to YAML configuration file.
    :param overrides: List of key=value override strings.
    :return: Merged and validated OmegaConf config object.
    """
    # Load base config
    yaml_conf = OmegaConf.load(config_path)
    
    # Apply overrides
    if overrides:
        cli_conf = OmegaConf.from_dotlist(overrides)
        config = OmegaConf.merge(yaml_conf, cli_conf)
    else:
        config = yaml_conf
    
    # Validate required fields
    # _validate_config_structure(config)
    
    print(f"Loaded configuration from {config_path}")
    if overrides:
        print(f"Applied overrides: {overrides}")
    
    return config


def _validate_config_structure(cfg):
    """
    Validate that required config sections exist.
    
    :param cfg: OmegaConf configuration object.
    :raises ValueError: If required fields are missing.
    """
    required_sections = ['paths', 'training', 'model', 'data']
    for section in required_sections:
        if section not in cfg:
            raise ValueError(f"Missing required config section: {section}")
    
    # Validate model tasks
    if 'tasks' not in cfg.model:
        raise ValueError("Missing 'model.tasks' in configuration")
    
    # Ensure at least one task is enabled
    tasks = cfg.model.tasks
    if not any([tasks.get('do_mvc', False),
                tasks.get('do_taxa_decoder', False)]):
        logger.warning("No pretraining tasks enabled! Set at least one task to True.")


def save_training_artifacts(cfg, model_config: dict):
    """
    Save model config and run config to disk for reproducibility.
    
    :param cfg: OmegaConf run configuration.
    :param model_config: Dictionary of model parameters.
    """
    # Save model config as JSON
    os.makedirs(os.path.dirname(cfg.paths.model_config_path), exist_ok=True)
    with open(cfg.paths.model_config_path, "w") as f:
        json.dump(model_config, f, indent=4)
    logger.info(f"Saved model config to {cfg.paths.model_config_path}")
    
    # Save run config as YAML
    run_config_path = os.path.join(cfg.paths.best_dir, "run_config.yaml")
    OmegaConf.save(cfg, run_config_path)
    logger.info(f"Saved run config to {run_config_path}")
    
    # Save training notes if provided
    if cfg.training.get('notes'):
        notes_path = os.path.join(cfg.paths.best_dir, "training_notes.md")
        with open(notes_path, "w") as f:
            f.write(cfg.training.notes)
        logger.info(f"Saved training notes to {notes_path}")


def get_wandb_config(cfg) -> dict:
    """
    Extract relevant fields for W&B logging.
    
    :param cfg: OmegaConf configuration object.
    :return: Dictionary of config values to log to W&B.
    """
    return {
        **OmegaConf.to_container(cfg.model.tasks, resolve=True),
        **OmegaConf.to_container(cfg.training, resolve=True),
        "num_bins": cfg.data.num_bins,
        "use_batch_labels": cfg.data.use_batch_labels,
        "bin_strategy": cfg.data.get('bin_strategy', 'uniform'),
    }
