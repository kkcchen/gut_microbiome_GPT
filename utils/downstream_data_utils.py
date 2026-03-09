import numpy as np
import anndata as ad
import scipy.sparse as sp
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from omegaconf import OmegaConf, DictConfig
from sklearn.preprocessing import normalize
from skbio.stats.composition import clr, closure, multi_replace
from trainers import logger

def normalize_embeddings(adata: ad.AnnData, method: str = "clr") -> np.ndarray:
    """
    apply normalization to the data in adata.X based on the specified method.
    current normalization include clr and log, returns anndata with normalized data in adata.X
    Args:
        adata: AnnData object containing the embeddings in adata.X
        method: Normalization method to apply. Options: "l2", "clr", "log"
    """    
    # Convert to dense if sparse
    if sp.issparse(adata.X):
        X = adata.X.toarray()
    else:
        X = adata.X.copy()
    
    if method == "l2":
        X_norm = normalize(X, norm='l2', axis=1)
    elif method == "clr":
        # Centered log-ratio transformation - for compositional data
        X_replaced = multi_replace(X)      # Replace zeros
        X_closed = closure(X_replaced)     # Closure (sum to 1)
        X_norm = clr(X_closed)
    elif method == "log":
        # Log transformation
        pseudocount = 1.0
        X_norm = np.log1p(X)# + pseudocount)  # log(1 + x)   
    elif method == "none":
        # No normalization
        X_norm = X
    elif method == "rel_ab":
        # No normalization
        X_norm = X + 1e-8
        X_norm = closure(X_norm)
    else:
        raise ValueError(
            f"Unknown normalization method: '{method}'. "
            f"Available methods: 'l2', 'clr', 'log', 'none'"
        )
    adata.X = X_norm
    return adata

def apply_thresholding(adata_train, adata_test, prevalence_threshold, abundance_threshold):
    """
    Apply prevalence and abundance thresholding to the embeddings.
    
    Args:
        adata_train: Training AnnData
        adata_test: Test AnnData
        prevalence_threshold: Minimum prevalence (fraction of samples) for a feature to be kept
        abundance_threshold: Minimum mean abundance for a feature to be kept
    Returns:
        Tuple of (filtered_train_adata, filtered_test_adata)
    """
    adata_train = adata_train.copy()
    adata_test = adata_test.copy()

    # combine train and test to compute thresholds
    combined_X = np.vstack([adata_train.X, adata_test.X])
    num_samples = combined_X.shape[0]
    prevalence = np.sum(combined_X > 0, axis=0) / num_samples
    mean_abundance = np.mean(combined_X, axis=0)
    # Identify features to keep
    features_to_keep = np.where(
        (prevalence >= prevalence_threshold) & 
        (mean_abundance >= abundance_threshold)
    )[0]
    logger.info(f"Thresholding: Keeping {len(features_to_keep)} features out of {combined_X.shape[1]}")
    # Subset the AnnData objects
    adata_train = adata_train[:, features_to_keep]
    adata_test = adata_test[:, features_to_keep]
    return adata_train, adata_test


def handle_normalization_and_thresholding(adata_train: ad.AnnData, adata_test: ad.AnnData, cfg: DictConfig) -> Tuple[ad.AnnData, ad.AnnData]:
    """
    Handle normalization and thresholding of embeddings based on config.
    
    Args:
        adata_train: Training AnnData
        adata_test: Test AnnData
        cfg: Configuration object

    Returns:
        Tuple of (normalized_train_adata, normalized_test_adata)
    """
    # apply thresholding if specified
    if cfg.downstream_tasks_config.get("prevalence_threshold", None) is not None and cfg.downstream_tasks_config.get("abundance_threshold", None):
        prevalence_threshold = cfg.downstream_tasks_config.prevalence_threshold
        abundance_threshold = cfg.downstream_tasks_config.abundance_threshold
        logger.info(f"Applying prevalence thresholding: {prevalence_threshold}, abundance threshold: {abundance_threshold}")
        adata_train, adata_test = apply_thresholding(adata_train, adata_test, prevalence_threshold, abundance_threshold)
    else:
        logger.warning("No thresholding specified, using all features")

    # check if normalization is specified in config
    if cfg.downstream_tasks_config.get('normalization', None) is not None:
        norm_method = cfg.downstream_tasks_config.normalization
        logger.info(f"Applying normalization: {norm_method}")
        adata_train = normalize_embeddings(adata_train, method=norm_method)
        adata_test = normalize_embeddings(adata_test, method=norm_method)
    else:
        logger.warning("No normalization specified, using raw embeddings")

    logger.info(f"After normalization and thresholding: Train shape: {adata_train.shape}, Test shape: {adata_test.shape}")

    return adata_train.copy(), adata_test.copy()