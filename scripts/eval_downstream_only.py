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
from utils.downstream_utils import run_downstream_evaluation

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

    logger.info("Running downstream tasks with extracted embeddings...")
    run_downstream_evaluation(cfg, accelerator, skip_if_exists=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run eval on downstream"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to evaluation YAML configuration file"
    )
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Override config values (e.g., checkpoint.path=/path/to/model.pt)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    cfg = load_and_validate_config(args.config, args.overrides)
    
    # Run evaluation
    main(cfg)
