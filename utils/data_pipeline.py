"""
Main data preparation pipeline orchestration.
"""
import anndata as ad
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from trainers import logger
from sklearn.model_selection import train_test_split
from data_utils import MicrobiomeDataset, MicrobiomeCollator, TaxaVocabulary, BatchVocabulary, build_tg_data_from_taxon_df


def prepare_microbiome_data(cfg, accelerator) -> Dict:
    """
    data pipeline for microbiome representation learning.
    
    1. Load preprocessed AnnData
    2. Split train/validation
    3. Build vocabularies
    4. Create datasets (raw data, no preprocessing, assume preprocessing is done)
    5. Dataset.__getitem__() handles dynamic top-k selection per access
    
    :param cfg: Configuration object.
    :param accelerator: Accelerator for distributed training.
    :return: Dictionary with dataloaders and vocabularies.
    """
    # 1. Load preprocessed AnnData (keep raw counts)
    logger.info(f"Loading preprocessed data from {cfg.paths.ann_table_path}")
    adata = load_anndata(cfg.paths.ann_table_path)
    
    # 2. Split train/validation
    logger.info("Splitting train/validation...")
    train_adata, valid_adata = split_data(
        adata,
        split_key=cfg.data.get('split_key', None),
        val_size=cfg.training.get('val_size', 0.1),
        seed=cfg.training.seed
    )
    
    # 3. Build vocabularies from training data
    logger.info("Building vocabularies...")
    taxa_vocab = TaxaVocabulary.from_adata(train_adata)
    batch_vocab = BatchVocabulary.from_adata(train_adata) if cfg.data.use_batch_labels else None
    
    # 4. Build taxonomic graph (if using GNN)
    graph_data = None
    if cfg.model.get('use_gnn', False):
        logger.info("Building taxonomic graph...")
        graph_data = build_tg_data_from_taxon_df(adata.varm['taxonomy'], taxa_vocab.vocab_list)
    
    # 5. Create datasets (raw data, no preprocessing)
    logger.info("Creating datasets with dynamic top-k selection...")
    train_dataset = MicrobiomeDataset(
        adata=train_adata,
        taxa_vocab=taxa_vocab,
        batch_vocab=batch_vocab,
        max_seq_len=cfg.data.max_seq_len,
        metadata_fields=cfg.data.get('metadata_fields', []),
    )
    
    valid_dataset = MicrobiomeDataset(
        adata=valid_adata,
        taxa_vocab=taxa_vocab,
        batch_vocab=batch_vocab,
        max_seq_len=cfg.data.max_seq_len,
        metadata_fields=cfg.data.get('metadata_fields', []),
    )
    
    # 6. Create dataloaders
    logger.info("Creating dataloaders...")
    if 'downsample_ratio_range' not in cfg.data:
        logger.warning("Config 'data.downsample_ratio_range' not found, using default: (0.5, 0.9)")
    if 'upsample_ratio_range' not in cfg.data:
        logger.warning("Config 'data.upsample_ratio_range' not found, using default: (1.1, 2.0)")
    if 'perturbation_ratio' not in cfg.data:
        logger.warning("Config 'data.perturbation_ratio' not found, using default: 0.6")
    if 'distribution' not in cfg.data:
        logger.warning("Config 'data.perturbation_distribution' not found, using default: 'zinb'")
    if 'perturbation_scale' not in cfg.data:
        logger.warning("Config 'data.perturbation_scale' not found, using default: 0.55")
    if 'norm_strategy' not in cfg.data:
        logger.warning("Config 'data.norm_strategy' not found, using default: clr")
    collator = MicrobiomeCollator(
        max_seq_len=cfg.data.max_seq_len,
        downsample_ratio_range=cfg.data.get('downsample_ratio_range', (0.5, 0.9)),
        upsample_ratio_range=cfg.data.get('upsample_ratio_range', (1.1, 2.0)),
        perturbation_ratio=cfg.data.get('perturbation_ratio', 0.6),
        perturbation_distribution=cfg.data.get('distribution', 'zinb'),
        perturbation_scale=cfg.data.get('perturbation_scale', 0.55),
        norm_strategy=cfg.data.get('norm_strategy', 'clr'),
        do_contrastive=cfg.model.tasks.get('do_contrastive', False)
    )
    
    if "num_workers" not in cfg.data:
        logger.warning("Config 'data.num_workers' not found, using default: 1")
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.get('num_workers', 1),
        collate_fn=collator,
        pin_memory=True,
        drop_last=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.validation.batch_size,
        shuffle=False,
        num_workers=cfg.data.get('num_workers', 1),
        collate_fn=collator,
        pin_memory=True
    )
    
    # 7. Print statistics
    if accelerator.is_main_process:
        print_data_statistics(train_adata, valid_adata, taxa_vocab, batch_vocab, cfg.data.max_seq_len)
    
    return {
        'train_loader': train_loader,
        'valid_loader': valid_loader,
        'taxa_vocab': taxa_vocab,
        'batch_vocab': batch_vocab,
        'graph_data': graph_data
    }


def load_anndata(file_path: str) -> ad.AnnData:
    """
    Load preprocessed AnnData from h5ad file.
    Keep raw counts - no transformations applied here.
    
    :param file_path: Path to .h5ad file.
    :return: AnnData object.
    """
    adata = ad.read_h5ad(file_path)
    # Validate structure:TODO
    return adata


def split_data(
    adata: ad.AnnData,
    split_key: Optional[str],
    val_size: float,
    seed: int
) -> Tuple[ad.AnnData, ad.AnnData]:
    """
    Split data into train/validation. If split_key is provided, use it to group samples. Data split should be entire studies.
    that is, for any study_id, all samples with that study_id go to either train or validation.
    
    :param adata: Full AnnData object.
    :param split_key: Column in adata.obs with split labels.
    :param val_size: Validation fraction.
    :param seed: Random seed.
    :return: Tuple of (train_adata, valid_adata).
    """
    if split_key is None:
        # Simple random split
        train_idx, valid_idx = train_test_split(
            range(adata.n_obs),
            test_size=val_size,
            random_state=seed
        )
    else:
        # Group by split_key
        # Ensure that all samples with the same group label go to either train or validation
        # the final train and valid sets should be as close as possible to the desired val_size
        # Group-level split: ensure all samples from same group go to same split
        if split_key not in adata.obs.columns:
            raise ValueError(f"split_key '{split_key}' not found in adata.obs columns")
        
        # Get unique groups and their sample counts
        group_ids = adata.obs[split_key].values
        unique_groups = np.unique(group_ids)
        
        # Count samples per group
        group_counts = {}
        group_indices = {}
        for group in unique_groups:
            indices = np.where(group_ids == group)[0]
            group_counts[group] = len(indices)
            group_indices[group] = indices
        
        shuffled_groups = unique_groups.copy()
        np.random.shuffle(shuffled_groups)
        total_samples = adata.n_obs
        target_val_samples = int(total_samples * val_size)
        val_groups = []
        val_sample_count = 0
        for group in shuffled_groups:
            if val_sample_count < target_val_samples:
                val_groups.append(group)
                val_sample_count += group_counts[group]
            else:
                if abs(val_sample_count + group_counts[group] - target_val_samples) < abs(val_sample_count - target_val_samples):
                    val_groups.append(group)
                    val_sample_count += group_counts[group]
        val_groups_set = set(val_groups)
        
        # Collect indices
        train_idx = []
        valid_idx = []
        
        for group in unique_groups:
            if group in val_groups_set:
                valid_idx.extend(group_indices[group])
            else:
                train_idx.extend(group_indices[group])
        
        train_idx = np.array(train_idx)
        valid_idx = np.array(valid_idx)
        
        # Log split statistics
        actual_val_ratio = len(valid_idx) / total_samples
        logger.info(f"Split by '{split_key}':")
        logger.info(f"  Total groups: {len(unique_groups)}")
        logger.info(f"  Train groups: {len(unique_groups) - len(val_groups)}")
        logger.info(f"  Val groups: {len(val_groups)}")
        logger.info(f"  Target val ratio: {val_size:.3f}")
        logger.info(f"  Actual val ratio: {actual_val_ratio:.3f}")
    train_adata = adata[train_idx].copy()
    valid_adata = adata[valid_idx].copy()
    
    return train_adata, valid_adata


def print_data_statistics(
    train_adata: ad.AnnData,
    valid_adata: ad.AnnData,
    vocab: TaxaVocabulary,
    batch_vocab: Optional[BatchVocabulary],
    max_seq_len: int
):
    """Print summary statistics."""
    logger.info("=" * 60)
    logger.info("DATA STATISTICS")
    logger.info("=" * 60)
    logger.info(f"Training samples: {train_adata.n_obs}")
    logger.info(f"Validation samples: {valid_adata.n_obs}")
    logger.info(f"Total taxa in vocabulary: {len(vocab)}")
    logger.info(f"Max sequence length (top-k + random): {max_seq_len}")
    
    if batch_vocab:
        logger.info(f"Number of batches/studies: {len(batch_vocab)}")
    
    # Statistics on expressed taxa per sample
    if hasattr(train_adata.X, 'toarray'):
        train_counts = train_adata.X.toarray()
    else:
        train_counts = train_adata.X
    
    n_expressed_per_sample = (train_counts > 0).sum(axis=1)
    logger.info(f"Avg expressed taxa per sample: {n_expressed_per_sample.mean():.1f}")
    logger.info(f"Min expressed taxa: {n_expressed_per_sample.min()}")
    logger.info(f"Max expressed taxa: {n_expressed_per_sample.max()}")
    logger.info(f"Samples with >= {max_seq_len} expressed: {(n_expressed_per_sample >= max_seq_len).sum()}")
    logger.info("=" * 60)