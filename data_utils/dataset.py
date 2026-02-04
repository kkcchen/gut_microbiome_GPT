"""
Dataset class with DYNAMIC top-k selection per epoch.
Selection happens in __getitem__(), so each epoch sees different unexpressed taxa.
"""
import numpy as np
import anndata as ad
from typing import Dict, List, Tuple, Optional
from torch.utils.data import Dataset

from .vocabs import TaxaVocabulary, BatchVocabulary
from trainers import logger


class MicrobiomeDataset(Dataset):
    """
    Dataset with dynamic top-k selection and random unexpressed sampling.
    
    Each call to __getitem__():
    1. Select ALL expressed taxa (count > 0)
    2. If n_expressed >= max_seq_len: randomly select max_seq_len from expressed
    3. If n_expressed < max_seq_len: take all expressed + randomly sample unexpressed
    """
    
    def __init__(
        self,
        adata: ad.AnnData,
        taxa_vocab: TaxaVocabulary,
        batch_vocab: Optional[BatchVocabulary],
        max_seq_len: int = 200,
        metadata_fields: Optional[List[str]] = None,
    ):
        """
        Initialize microbiome dataset.
        
        :param adata: AnnData object with count data in .X (RAW counts).
        :param taxa_vocab: Taxa vocabulary.
        :param batch_vocab: Batch vocabulary (optional).
        :param max_seq_len: Fixed sequence length per sample.
        """
        self.adata = adata
        self.taxa_vocab = taxa_vocab
        self.batch_vocab = batch_vocab
        self.max_seq_len = max_seq_len
        
        # Convert sparse to dense if needed for efficient indexing
        if hasattr(adata.X, 'toarray'):
            self.counts = adata.X.toarray()
        else:
            self.counts = adata.X
        
        # Pre-compute taxa IDs for vocabulary encoding
        self.all_taxa_ids = taxa_vocab.encode_taxa(adata.var_names.tolist())
        
        # For batch labels
        self.use_batch_labels = batch_vocab is not None
        if self.use_batch_labels:
            self.batch_ids = batch_vocab.encode_batches(
                adata.obs['study_id'].tolist()
            )
        
        # for metadata in the adata obs fields
        self.metadata_fields = metadata_fields if metadata_fields else []
        if len(self.metadata_fields) > 0:
            for field in self.metadata_fields:
                if field not in adata.obs.columns:
                    raise ValueError(f"Metadata field '{field}' not found in adata.obs")
        else:
            logger.warning("No metadata fields specified for dataset.")

    
    def __len__(self) -> int:
        """Return number of samples."""
        return self.adata.X.shape[0]
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample with top-k selection.
                
        Returns:
        - taxa_ids: Selected taxa indices (max_seq_len,)
        - original_counts: Original count values (max_seq_len,)
        - expressed_mask: Boolean mask for truly expressed taxa (max_seq_len,)
        - original_depth: Original total count
        - batch_id: Batch/study ID (if using batch labels)
        - metadata: Additional metadata fields from adata.obs
        
        :param idx: Sample index.
        :return: Dictionary with sample data.
        """
        # Get original counts for this sample (FULL vector)
        original_counts = self.counts[idx]  # (n_total_taxa,)
        original_depth = original_counts.sum()
        
        # sample taxa
        selected_indices, selected_taxa_ids, selected_counts, expressed_mask, original_depth = \
            self._select_taxa_dynamic(original_counts)
        
        # Build output dictionary
        sample_dict = {
            'taxa_ids': selected_taxa_ids,  # Selected taxa IDs
            'original_counts': selected_counts,  # Selected counts
            'expressed_mask': expressed_mask,  # Selected mask
            'original_depth': original_depth,
        }
        
        if self.use_batch_labels:
            sample_dict['batch_id'] = self.batch_ids[idx]
        
        if len(self.metadata_fields) > 0:
            for field in self.metadata_fields:
                sample_dict[field] = self.adata.obs.iloc[idx][field]
        
        return sample_dict
    # TODO: IMPORTANT - NOTE THIS IS DONE BEFORE THE COLLATOR
    # SO CLR ETC AND DOWNSAMPLING ARE COMPUTED ON THE SELECTED TAXA ONLY
    def _select_taxa_dynamic(
        self,
        counts: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Dynamically select taxa: expressed + random unexpressed.
        the assumption is that max_seq_len is large enough that most samples dont have this number of expressed taxa
                
        Strategy:
        1. Find all expressed taxa (count > 0)
        2. If n_expressed >= max_seq_len:
           - select top max_seq_len most expressed taxa
        3. If n_expressed < max_seq_len:
           - Take all expressed taxa
           - Randomly sample (max_seq_len - n_expressed) from unexpressed taxa
        
        :param counts: Count vector for sample (n_total_taxa,).
        :return: Tuple of (selected_indices, selected_taxa_ids, selected_counts, expressed_mask).
        """
        n_total_taxa = len(counts)
        
        # Identify expressed and unexpressed taxa
        expressed_mask_full = counts > 0
        expressed_indices = np.where(expressed_mask_full)[0]
        unexpressed_indices = np.where(~expressed_mask_full)[0]
        original_depth = counts.sum()
        
        n_expressed = len(expressed_indices)
        
        if n_expressed >= self.max_seq_len:
            # Case 1: Enough expressed taxa: select top max_seq_len
            top_k_local_indices = np.argsort(counts[expressed_indices])[-self.max_seq_len:]
            selected_indices = expressed_indices[top_k_local_indices]

            # All selected are expressed
            expressed_mask = np.ones(self.max_seq_len, dtype=bool)
        
        else:
            # Case 2: Not enough expressed, need to add unexpressed
            n_unexpressed_needed = self.max_seq_len - n_expressed
            
            # random sampling of unexpressed taxa (varies each epoch)
            sampled_unexpressed = np.random.choice(
                unexpressed_indices,
                size=min(n_unexpressed_needed, len(unexpressed_indices)),
                replace=False
            )
            # TODO: consider having this weighted by the taxa observed frequency in the dataset
            
            # Combine expressed + unexpressed
            selected_indices = np.concatenate([expressed_indices, sampled_unexpressed])
            
            # Create mask: True for expressed, False for unexpressed
            expressed_mask = np.concatenate([
                np.ones(n_expressed, dtype=bool),
                np.zeros(len(sampled_unexpressed), dtype=bool)
            ])

        # shuffle selected indices to mix expressed and unexpressed
        perm = np.random.permutation(len(selected_indices))
        selected_indices = selected_indices[perm]
        expressed_mask = expressed_mask[perm]
        
        # Get taxa IDs and counts for selected indices
        selected_taxa_ids = self.all_taxa_ids[selected_indices]
        selected_counts = counts[selected_indices]
        
        return selected_indices, selected_taxa_ids, selected_counts, expressed_mask, original_depth
