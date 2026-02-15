"""
Main data preparation pipeline orchestration.
"""
import torch
import anndata as ad
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from typing import Dict, Tuple, Optional
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from trainers import logger
from sklearn.model_selection import train_test_split
from data_utils import MicrobiomeDataset, MicrobiomeCollator, TaxaVocabulary, BatchVocabulary, FinetuningDataset, build_tg_data_from_taxon_df


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
    
    if cfg.debug.get('nrows', None) is not None:
        adata = adata[:cfg.debug.nrows].copy()
        logger.warning(f"Debug mode: using only {cfg.debug.nrows} rows")
    
    # 2. Split train/validation
    logger.info("Splitting train/validation...")
    train_adata, valid_adata = split_data(
        adata,
        split_key=cfg.data.get('split_key', None),
        val_size=cfg.data.get('val_size', 0.1),
        seed=cfg.training.seed
    )
    
    # 3. Build vocabularies from training data
    logger.info("Building vocabularies...")
    taxa_vocab = TaxaVocabulary.from_adata(train_adata)
    batch_vocab = BatchVocabulary.from_adata(train_adata) if cfg.data.use_batch_labels else None
    taxa_vocab.save(cfg.paths.taxa_vocab_path)
    if batch_vocab:
        batch_vocab.save(cfg.paths.batch_vocab_path)
        logger.info(f"Batch Vocab saved at {cfg.paths.batch_vocab_path} with {len(batch_vocab)} batches")
    logger.info(f"Taxa Vocab saved at {cfg.paths.taxa_vocab_path} with {len(taxa_vocab)} taxa")
    
    # 4. Build taxonomic graph (if using GNN) take a look here
    graph_data = None
    if cfg.model.params.get('use_gnn', False):
        logger.info("Building taxonomic graph...")
        graph_data = build_tg_data_from_taxon_df(adata.varm['taxonomy'], taxa_vocab.id_to_token)
        torch.save(graph_data, cfg.paths.graph_path)
    
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
    if 'downsample_distribution' not in cfg.data:
        logger.warning("Config 'data.downsample_distribution' not found, using default: 'binomial'")
    if 'perturbation_scale' not in cfg.data:
        logger.warning("Config 'data.perturbation_scale' not found, using default: 0.55")
    if 'norm_strategy' not in cfg.data:
        logger.warning("Config 'data.norm_strategy' not found, using default: clr")
    collator = MicrobiomeCollator(
        max_seq_len=cfg.data.max_seq_len,
        downsample_ratio_range=cfg.data.get('downsample_ratio_range', (0.5, 0.9)),
        upsample_ratio_range=cfg.data.get('upsample_ratio_range', (1.1, 2.0)),
        perturbation_ratio=cfg.data.get('perturbation_ratio', 0.6),
        perturbation_distribution=cfg.data.get('downsample_distribution', 'binomial'),
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


def prepare_inference_data(cfg, input_file, accelerator) -> Dict:
    """
    Data pipeline for microbiome inference.
    
    1. Load preprocessed AnnData
    2. Load existing vocabularies (built during training)
    3. Create dataset (no augmentation)
    4. Create dataloader (no shuffling, no perturbations)
    
    :param cfg: Configuration object (inference config).
    :param input_file: Path to the input data file for inference.
    :param accelerator: Accelerator for distributed inference.
    :return: Dictionary with dataloader, vocabularies, and graph data.
    """
    # 1. Load preprocessed AnnData (keep raw counts)
    logger.info(f"Loading inference data from {input_file}")
    adata = load_anndata(input_file)
    
    logger.info(f"Loaded {adata.n_obs} samples with {adata.n_vars} taxa")
    
    # 2. Load vocabularies (must exist from training)
    logger.info("Loading vocabularies from training...")
    
    if not Path(cfg.paths.taxa_vocab_path).exists():
        raise FileNotFoundError(
            f"Taxa vocabulary not found at {cfg.paths.taxa_vocab_path}. "
            "Please run training first to generate vocabularies."
        )
    
    taxa_vocab = TaxaVocabulary.load(cfg.paths.taxa_vocab_path)
    logger.info(f"Loaded taxa vocabulary: {len(taxa_vocab)} taxa")
    
    batch_vocab = None
    if cfg.data.use_batch_labels:
        if not Path(cfg.paths.batch_vocab_path).exists():
            raise FileNotFoundError(
                f"Batch vocabulary not found at {cfg.paths.batch_vocab_path}. "
                "Please run training first to generate vocabularies."
            )
        batch_vocab = BatchVocabulary.load(cfg.paths.batch_vocab_path)
        logger.info(f"Loaded batch vocabulary: {len(batch_vocab)} batches (includes <UNK>)")
    
    # 3. Load taxonomic graph (if using GNN)
    graph_data = None
    if cfg.model.get('use_gnn', False):
        if cfg.paths.get('graph_path') and Path(cfg.paths.graph_path).exists():
            logger.info(f"Loading taxonomic graph from {cfg.paths.graph_path}")
            graph_data = torch.load(cfg.paths.graph_path)
        else:
            logger.warning("GNN enabled but graph_path not found. Building graph from data...")
            graph_data = build_tg_data_from_taxon_df(adata.varm['taxonomy'], taxa_vocab.vocab_list)
    
    # 4. Create dataset (no augmentation for inference)
    logger.info("Creating inference dataset...")
    inference_dataset = MicrobiomeDataset(
        adata=adata,
        taxa_vocab=taxa_vocab,
        batch_vocab=batch_vocab,
        max_seq_len=cfg.data.max_seq_len,
        metadata_fields=cfg.data.get('metadata_fields', []),
    )
    
    # 5. Create collator for inference (NO perturbations)
    logger.info("Creating inference collator (no perturbations)...")
    inference_collator = MicrobiomeCollator(
        max_seq_len=cfg.data.max_seq_len,
        downsample_ratio_range=None,
        upsample_ratio_range=None,
        perturbation_ratio=0.0,
        perturbation_distribution=None,
        perturbation_scale=0.0,
        eval_mode=True,
    )
    
    # 6. Create dataloader
    logger.info("Creating inference dataloader...")
    if "num_workers" not in cfg.data:
        logger.warning("Config 'data.num_workers' not found, using default: 1")
    
    if "batch_size" not in cfg.data:
        logger.warning("Config 'data.batch_size' not found, using default: 64")
    
    inference_loader = DataLoader(
        inference_dataset,
        batch_size=cfg.data.get('batch_size', 64),
        shuffle=False,  # Never shuffle for inference
        num_workers=cfg.data.get('num_workers', 1),
        collate_fn=inference_collator,
        pin_memory=True,
        drop_last=False  # Keep all samples
    )
    
    # 7. Print statistics
    if accelerator.is_main_process:
        print_inference_statistics(adata, taxa_vocab, batch_vocab, cfg.data.max_seq_len)
    
    return {
        'inference_loader': inference_loader,
        'taxa_vocab': taxa_vocab,
        'batch_vocab': batch_vocab,
        'graph_data': graph_data,
        'adata': adata,  # Return adata for metadata access
    }


def print_inference_statistics(adata, taxa_vocab, batch_vocab, max_seq_len):
    """
    Print statistics about inference data.
    
    :param adata: AnnData object.
    :param taxa_vocab: TaxaVocabulary.
    :param batch_vocab: BatchVocabulary (optional).
    :param max_seq_len: Maximum sequence length.
    """
    logger.info("=" * 80)
    logger.info("INFERENCE DATA STATISTICS")
    logger.info("=" * 80)
    
    # Sample statistics
    logger.info(f"Number of samples: {adata.n_obs}")
    logger.info(f"Number of taxa: {adata.n_vars}")
    
    # Vocabulary statistics
    logger.info(f"Taxa vocabulary size: {len(taxa_vocab)}")
    if batch_vocab is not None:
        logger.info(f"Batch vocabulary size: {len(batch_vocab)} (includes <UNK>)")
    
    # Sequencing depth statistics
    depths = adata.X.sum(axis=1)
    if hasattr(depths, 'A1'):  # Sparse matrix
        depths = depths.A1
    logger.info(f"Sequencing depth: min={depths.min():.0f}, "
                f"median={np.median(depths):.0f}, "
                f"max={depths.max():.0f}")
    
    # Taxa per sample statistics
    expressed_per_sample = (adata.X > 0).sum(axis=1)
    if hasattr(expressed_per_sample, 'A1'):  # Sparse matrix
        expressed_per_sample = expressed_per_sample.A1
    logger.info(f"Expressed taxa per sample: "
                f"min={expressed_per_sample.min()}, "
                f"median={np.median(expressed_per_sample):.0f}, "
                f"max={expressed_per_sample.max()}")
    
    # Check if any samples exceed max_seq_len
    samples_exceeding = (expressed_per_sample > max_seq_len).sum()
    if samples_exceeding > 0:
        logger.warning(
            f"{samples_exceeding} samples ({100*samples_exceeding/adata.n_obs:.1f}%) "
            f"have more than {max_seq_len} expressed taxa. "
            f"Top-{max_seq_len} selection will be applied."
        )
    
    # Batch information
    if batch_vocab is not None and 'study_id' in adata.obs:
        unique_batches = adata.obs['study_id'].nunique()
        logger.info(f"Unique batches in data: {unique_batches}")
    
    logger.info("=" * 80)


def check_unknown_batches(adata, batch_vocab):
    """
    Check which batches in data are not in vocabulary (will be mapped to <UNK>).
    
    :param adata: AnnData object.
    :param batch_vocab: BatchVocabulary.
    :return: Set of unknown batch names.
    """
    if 'study_id' not in adata.obs:
        return set()
    
    data_batches = set(adata.obs['study_id'].unique())
    vocab_batches = set(batch_vocab.batch_names) - {'<UNK>'}  # Exclude <UNK> token
    
    unknown_batches = data_batches - vocab_batches
    
    return unknown_batches

def save_embeddings(
        embeddings_dict: Dict[str, torch.Tensor],
        original_adata: ad.AnnData,
        save_path: str,
    ):
        """
        Save extracted embeddings to disk.
        
        :param embeddings_dict: Dictionary returned from inference().
        :param original_adata: Original AnnData object.
        :param save_path: Path to save embeddings file.
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # Save in AnnData format for compatibility with downstream tasks
        # original adata is preserved in obsm named "raw"
        # new embedding is stored in obsm named "embedding"
        embedding_adata = ad.AnnData(
            X=embeddings_dict['embeddings'].cpu().numpy(),
            obs=original_adata.obs.copy(),
        )
        embedding_file = save_path
        # print statistics about the final adata, print obsm shapes for each obsm, obs columns, var
        logger.info(f"Final embedding AnnData shape: {embedding_adata.shape}")
        logger.info(f"obs columns: {list(embedding_adata.obs.columns)}")

        embedding_adata.write_h5ad(embedding_file)
        logger.info(f"Embeddings saved to {embedding_file}")

def prepare_finetune_data(cfg, accelerator):
    """
    Prepare finetuning data loaders with train/val split.
    
    :param cfg: Configuration object.
    :param accelerator: Accelerator instance.
    :return: Dictionary with data loaders and vocabularies.
    """
    # load train
    logger.info(f"Loading training data from: {cfg.paths.downstream_train}")
    # Load anndata
    adata_train = ad.read_h5ad(cfg.paths.downstream_train)
    logger.info(f"[Data Preparation] Loaded {adata_train.shape[0]} samples with {adata_train.shape[1]} taxa")

    # load test (optional)
    if hasattr(cfg.paths, 'downstream_test') and cfg.paths.downstream_test:
        logger.info(f"[Data Preparation] Loading test data from: {cfg.paths.downstream_test}")
        adata_test = ad.read_h5ad(cfg.paths.downstream_test)
        logger.info(f"[Data Preparation] Loaded {adata_test.shape[0]} test samples")
    else:
        logger.warning("[Data Preparation] No test file provided, skipping test evaluation")
        adata_test = None       
    
    # make copy and get the relevant chunk
    adata_train_task = adata_train[adata_train.obs['downstream_task'] == cfg.data.finetune_task_name].copy()
    adata_test_task = adata_test[adata_test.obs['downstream_task'] == cfg.data.finetune_task_name].copy() if adata_test is not None else None
    # Extract and process labels
    label_column = cfg.data.label_column
    if label_column not in adata_train_task.obs.columns:
        raise ValueError(f"Label column '{label_column}' not found in adata_train_task.obs")
    
    labels = adata_train_task.obs[label_column].values

    # 2. Load vocabularies (must exist from training)
    logger.info("Loading vocabularies from training...")
    
    if not Path(cfg.paths.taxa_vocab_path).exists():
        raise FileNotFoundError(
            f"Taxa vocabulary not found at {cfg.paths.taxa_vocab_path}. "
            "Please run training first to generate vocabularies."
        )
    
    taxa_vocab = TaxaVocabulary.load(cfg.paths.taxa_vocab_path)
    logger.info(f"Loaded taxa vocabulary: {len(taxa_vocab)} taxa")


    # # Create vocabularies
    # logger.info("[Data Preparation] Building vocabularies...")
    # taxa_vocab = TaxaVocabulary.from_adata(adata_train_task)
    # logger.info(f"[Data Preparation] Taxa vocabulary size: {len(taxa_vocab)}")
    
    batch_vocab = None
    if cfg.data.use_batch_labels:
        batch_vocab = BatchVocabulary()
        batch_vocab.build_vocab(adata_train_task.obs['study_id'].tolist())
        logger.info(f"[Data Preparation] Batch vocabulary size: {len(batch_vocab)}")

    # GNN data (optional)
    graph_data = None
    if cfg.model.params.get('use_gnn', False):
        if cfg.paths.get('graph_path') and Path(cfg.paths.graph_path).exists():
            logger.info(f"Loading taxonomic graph from {cfg.paths.graph_path}")
            graph_data = torch.load(cfg.paths.graph_path)
        else:
            logger.warning("GNN enabled but graph_path not found. Building graph from data...")
            graph_data = build_tg_data_from_taxon_df(adata_train.varm['taxonomy'], taxa_vocab.id_to_token)

    # Handle classification vs regression
    if cfg.training.finetune_task == 'classification':
        # Encode categorical labels
        le = LabelEncoder()
        labels_encoded = le.fit_transform(labels)
        adata_train_task.obs[label_column] = labels_encoded
        num_classes = len(le.classes_)
        logger.info(f"[Data Preparation] Classification task with {num_classes} classes: {le.classes_}")
        if adata_test_task is not None:
            adata_test_task.obs[label_column] = le.transform(adata_test_task.obs[label_column].values)
    else:
        # Regression - ensure numeric
        labels_encoded = labels.astype(np.float32)
        adata_train_task.obs[label_column] = labels_encoded
        num_classes = None
        logger.info(f"[Data Preparation] Regression task - label range: [{labels.min():.3f}, {labels.max():.3f}]")
        if adata_test_task is not None:
            adata_test_task.obs[label_column] = adata_test_task.obs[label_column].astype(np.float32)
    
    # Train/validation split
    train_idx, val_idx = train_test_split(
        np.arange(len(adata_train_task)),
        test_size=cfg.data.val_size,
        random_state=cfg.training.seed,
        stratify=labels_encoded if cfg.training.finetune_task == 'classification' else None
    )
    
    logger.info(f"[Data Preparation] Train samples: {len(train_idx)}, Validation samples: {len(val_idx)}")
    

    
    # Create datasets
    train_dataset = FinetuningDataset(
        adata=adata_train_task[train_idx],
        taxa_vocab=taxa_vocab,
        batch_vocab=batch_vocab,
        label_column=label_column,
        max_seq_len=cfg.data.max_seq_len,
        metadata_fields=cfg.data.get('metadata_fields', None)
    )
    
    val_dataset = FinetuningDataset(
        adata=adata_train_task[val_idx],
        taxa_vocab=taxa_vocab,
        batch_vocab=batch_vocab,
        label_column=label_column,
        max_seq_len=cfg.data.max_seq_len,
        metadata_fields=cfg.data.get('metadata_fields', None)
    )

    if adata_test_task is not None:
        test_dataset = FinetuningDataset(
            adata=adata_test_task,
            taxa_vocab=taxa_vocab,
            batch_vocab=batch_vocab,
            label_column=label_column,
            max_seq_len=cfg.data.max_seq_len,
            metadata_fields=cfg.data.get('metadata_fields', None)
        )
    
    # Create collator
    collator = MicrobiomeCollator(
        max_seq_len=cfg.data.max_seq_len,
        norm_strategy=cfg.data.norm_strategy,
        finetune_mode=True,  # Skip perturbation
        do_contrastive=False,
        eval_mode=False
    )

    eval_collator = MicrobiomeCollator(
        max_seq_len=cfg.data.max_seq_len,
        norm_strategy=cfg.data.norm_strategy,
        finetune_mode=True,  # Skip perturbation
        do_contrastive=False,
        eval_mode=True  # No augmentation, deterministic
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=cfg.data.get('num_workers', 0),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        collate_fn=eval_collator,
        num_workers=cfg.data.get('num_workers', 0),
        pin_memory=True
    )

    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            collate_fn=eval_collator,
            num_workers=cfg.data.get('num_workers', 0),
            pin_memory=True
        )
    else:
        test_loader = None
    
    return {
        'train_loader': train_loader,
        'valid_loader': val_loader,
        'test_loader': test_loader,
        'taxa_vocab': taxa_vocab,
        'batch_vocab': batch_vocab,
        'num_classes': num_classes,
        'graph_data': graph_data  # TODO: Add graph support if needed
    }
