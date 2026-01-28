"""
Model configuration and initialization utilities.
"""
import json
from omegaconf import OmegaConf
from trainers import logger


def build_model_config(cfg, taxa_vocab, batch_vocab=None, graph_data=None) -> dict:
    """
    Construct complete model configuration from training config and data artifacts.
    
    :param cfg: OmegaConf configuration object.
    :param taxa_vocab: Microbiome vocabulary object.
    :param batch_vocab: Batch vocabulary (if using batch labels).
    :param graph_data: Graph data object (if using GNN).
    :return: Complete model configuration dictionary.
    """
    # Start with static model parameters from config
    model_config = OmegaConf.to_container(cfg.model.params, resolve=True)
    
    # Add dynamic configuration based on data and tasks
    dynamic_config = {
        # Task flags
        "do_mvc": cfg.model.tasks.do_mvc,
        "do_taxa_decoder": cfg.model.tasks.do_taxa_decoder,
        "do_contrastive": cfg.model.tasks.do_contrastive,
        
        # Data configuration
        "use_batch_labels": cfg.data.use_batch_labels,
        "use_gnn": cfg.model.get('use_gnn', False),
        "n_input_bins": cfg.data.num_bins,
        
        # Vocabulary parameters
        "vocab_len": len(taxa_vocab),
        
        # Batch labels
        "num_batch_labels": len(batch_vocab) if batch_vocab else 0,
        
        # Paths
        "init_vocab_path": cfg.paths.get('vocab_path', None),
        
        # GNN parameters
        "num_gnn_nodes": graph_data.num_nodes if graph_data else None,
    }
    
    model_config.update(dynamic_config)
    
    logger.info("Model configuration built:")
    logger.info(json.dumps(model_config, indent=2))
    
    return model_config


def initialize_training_components(model_config: dict, cfg, total_steps: int):
    """
    Initialize model, optimizer, and scheduler.
    NOTE: This is a placeholder - actual implementation in checkpoint_utils
    or keep in train_functions.py
    
    :param model_config: Complete model configuration.
    :param cfg: Training configuration.
    :param total_steps: Total training steps for scheduler.
    :return: Tuple of (model, optimizer, scheduler).
    """
    # YOUR IMPLEMENTATION - Initialize fresh components
    # This would call your model class constructor
    pass
