"""
Microbiome Representation Learning Inference Script
Extract embeddings from trained models for downstream analysis.
"""
import argparse
import torch
import numpy as np
import os
from pathlib import Path
from omegaconf import OmegaConf
from accelerate import Accelerator

from utils.config_utils import load_and_validate_config
from utils.model_utils import build_model_config, load_trained_model, inference
from utils.data_pipeline import prepare_inference_data, save_embeddings
from trainers.trainer import MicrobiomeTrainer
from trainers import logger
from data_utils.vocabs import TaxaVocabulary, BatchVocabulary
from utils.checkpoint_utils import setup_directories_eval

def setup_inference_environment(cfg):
    """
    Initialize inference environment.
    
    :param cfg: OmegaConf configuration object.
    :return: Initialized Accelerator instance.
    """
    torch.manual_seed(cfg.eval.seed)
    np.random.seed(cfg.eval.seed)
    
    fp16 = cfg.eval.get('enable_fp16', False)
    accelerator = Accelerator(
        mixed_precision="fp16" if fp16 else "no",
    )
    
    # Setup directories and handle start_over flag
    cfg = setup_directories_eval(cfg, accelerator)

    return cfg, accelerator


def main(cfg):
    """
    Main inference workflow.
    
    :param cfg: OmegaConf configuration object.
    """
    cfg, accelerator = setup_inference_environment(cfg)
    
    logger.info("=" * 80)
    logger.info("MICROBIOME REPRESENTATION LEARNING - INFERENCE MODE")
    logger.info("=" * 80)

    # Prepare inference data
    logger.info("Preparing inference data...")
    inference_data_artifacts = prepare_inference_data(
        cfg=cfg,
        accelerator=accelerator
    )
    
    logger.info(f"Number of samples: {len(inference_data_artifacts['inference_loader'].dataset)}")
    logger.info(f"Number of batches: {len(inference_data_artifacts['inference_loader'])}")
    
    # Build model configuration
    logger.info("Building model configuration...")
    model_config = build_model_config(
        cfg=cfg,
        taxa_vocab=inference_data_artifacts['taxa_vocab'],
        batch_vocab=inference_data_artifacts['batch_vocab'],
        graph_data=inference_data_artifacts['graph_data'],
        eval=True
    )
    
    # Load trained model
    logger.info("Loading trained model...")
    model = load_trained_model(cfg, model_config, accelerator)
    
    # Move graph data to device
    if inference_data_artifacts['graph_data'] is not None:
        inference_data_artifacts['graph_data'] = inference_data_artifacts['graph_data'].to(accelerator.device)
    
    # Prepare inference loader
    inference_loader = accelerator.prepare(inference_data_artifacts['inference_loader'])
    
    # Initialize trainer for inference utilities
    # trainer = MicrobiomeTrainer(
    #     cfg=cfg,
    #     accelerator=accelerator,
    #     taxa_vocab=inference_data_artifacts['taxa_vocab'],
    #     batch_vocab=inference_data_artifacts['batch_vocab'],
    #     graph_data=inference_data_artifacts['graph_data']
    # )
    
    # Run inference
    logger.info("=" * 80)
    logger.info("Running inference to extract embeddings...")
    logger.info("=" * 80)
    
    results = inference(
        model=model,
        data_loader=inference_loader,
        graph_data=inference_data_artifacts['graph_data'],
        embedding_type=cfg.eval.embedding_type,
        return_outputs=cfg.eval.return_full_output
    )
    
    # Save results
    if accelerator.is_main_process:
        output_dir = Path(cfg.paths.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings
        if cfg.eval.save_embeddings:
            logger.info(f"Saving embeddings to inside {cfg.paths.output_dir}")
            save_path = output_dir / cfg.paths.save_name
            save_embeddings(
                embeddings_dict=results,
                original_adata=inference_data_artifacts['adata'],
                save_path=save_path,
            )
        
        # Save metadata
        if cfg.eval.save_metadata:
            metadata = {
                'num_samples': results['embeddings'].shape[0],
                'embedding_dim': results['embeddings'].shape[-1],
                'embedding_type': cfg.eval.embedding_type,
                'checkpoint_path': str(cfg.paths.checkpoint_path) if cfg.paths.checkpoint_path else cfg.checkpoint.load_from,
                'config': OmegaConf.to_container(cfg, resolve=True)
            }
            
            metadata_path = Path(cfg.paths.output_dir) / "inference_metadata.yaml"
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
