"""
Microbiome Representation Learning Inference Script
Extract embeddings from trained models for downstream analysis.
"""
import argparse
import torch
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from accelerate import Accelerator

from utils.config_utils import load_and_validate_config
from utils.model_utils import build_model_config, load_trained_model
from utils.data_pipeline import prepare_inference_data
from trainers.trainer import MicrobiomeTrainer
from trainers import logger
from data_utils.vocabs import TaxaVocabulary, BatchVocabulary

def setup_inference_environment(cfg):
    """
    Initialize inference environment.
    
    :param cfg: OmegaConf configuration object.
    :return: Initialized Accelerator instance.
    """
    torch.manual_seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)
    
    accelerator = Accelerator(
        mixed_precision="fp16" if cfg.training.enable_fp16 else "no",
    )
    
    return accelerator


def main(cfg):
    """
    Main inference workflow.
    
    :param cfg: OmegaConf configuration object.
    """
    accelerator = setup_inference_environment(cfg)
    
    logger.info("=" * 80)
    logger.info("MICROBIOME REPRESENTATION LEARNING - INFERENCE MODE")
    logger.info("=" * 80)

    # Prepare inference data
    logger.info("Preparing inference data...")
    inference_loader = prepare_inference_data(
        cfg=cfg,
        accelerator=accelerator
    )
    
    logger.info(f"Number of samples: {len(inference_loader.dataset)}")
    logger.info(f"Number of batches: {len(inference_loader)}")
    
    # Build model configuration
    logger.info("Building model configuration...")
    model_config = build_model_config(
        cfg=cfg,
        taxa_vocab=inference_loader['taxa_vocab'],
        batch_vocab=inference_loader['batch_vocab'],
        graph_data=inference_loader['graph_data']
    )
    
    # Load trained model
    logger.info("Loading trained model...")
    model = load_trained_model(cfg, model_config, accelerator)
    
    # Move graph data to device
    if inference_loader['graph_data'] is not None:
        inference_loader['graph_data'] = inference_loader['graph_data'].to(accelerator.device)
    
    # Prepare inference loader
    inference_loader = accelerator.prepare(inference_loader)
    
    # Initialize trainer for inference utilities
    trainer = MicrobiomeTrainer(
        cfg=cfg,
        accelerator=accelerator,
        taxa_vocab=inference_loader['taxa_vocab'],
        batch_vocab=inference_loader['batch_vocab'],
        graph_data=inference_loader['graph_data']
    )
    
    # Run inference
    logger.info("=" * 80)
    logger.info("Running inference to extract embeddings...")
    logger.info("=" * 80)
    
    results = trainer.inference(
        model=model,
        data_loader=inference_loader,
        embedding_type=cfg.inference.embedding_type,
        return_outputs=cfg.inference.return_full_output
    )
    
    # Save results
    if accelerator.is_main_process:
        output_dir = Path(cfg.inference.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings
        if cfg.inference.save_embeddings:
            embeddings_path = Path(cfg.paths.embeddings_file)
            logger.info(f"Saving embeddings to {embeddings_path}")
            trainer.save_embeddings(
                embeddings_dict=results,
                save_path=embeddings_path,
                format=cfg.inference.output_format
            )
        
        # Save metadata
        if cfg.inference.save_metadata:
            metadata = {
                'num_samples': results['embeddings'].shape[0],
                'embedding_dim': results['embeddings'].shape[-1],
                'embedding_type': cfg.inference.embedding_type,
                'checkpoint_path': str(cfg.checkpoint.path) if cfg.checkpoint.path else cfg.checkpoint.load_from,
                'config': OmegaConf.to_container(cfg, resolve=True)
            }
            
            metadata_path = Path(cfg.paths.metadata_file)
            logger.info(f"Saving metadata to {metadata_path}")
            OmegaConf.save(metadata, metadata_path)
    
    logger.info("=" * 80)
    logger.info("INFERENCE COMPLETE!")
    logger.info(f"Extracted {results['embeddings'].shape[0]} embeddings")
    logger.info(f"Embedding dimension: {results['embeddings'].shape[-1]}")
    logger.info("=" * 80)
    
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference with trained microbiome transformer model"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to inference YAML configuration file"
    )
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Override config values (e.g., checkpoint.path=/path/to/model.pt)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    cfg = load_and_validate_config(args.config, args.overrides)
    
    # Run inference
    main(cfg)
