"""
Collate function for microbiome abundance data.
"""
import torch
import numpy as np
from typing import List, Dict
from skbio.stats.composition import closure


class MicrobiomeCollator:
    """
    Collator that batches variable-length taxa/count sequences into padded
    tensors and applies normalization directly to the original counts.
    """

    def __init__(
        self,
        max_seq_len: int = 200,
        norm_strategy: str = 'clr',
        num_bins: int = 15,
    ):
        """
        Initialize collator with normalization parameters.

        :param max_seq_len: Number of taxa to select.
        :param norm_strategy: Normalization strategy applied to counts.
        :param num_bins: Number of quantile bins used when norm_strategy is 'binning'.
        """
        self.max_seq_len = max_seq_len
        self.norm_strategy = norm_strategy
        self.num_bins = num_bins

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate with padding for variable-length sequences.
        """
        from torch.nn.utils.rnn import pad_sequence

        batch_size = len(batch)

        # Extracts lists of variable-length arrays
        taxa_ids_list = [torch.from_numpy(s['taxa_ids']).long() for s in batch]
        counts_list = [torch.from_numpy(s['original_counts']).float() for s in batch]
        expressed_mask_list = [torch.from_numpy(s['expressed_mask']).bool() for s in batch]
        depths_original = torch.tensor([s['original_depth'] for s in batch]).float()

        # Pad sequences
        taxa_ids_padded = pad_sequence(taxa_ids_list, batch_first=True, padding_value=0)
        counts_padded = pad_sequence(counts_list, batch_first=True, padding_value=0)
        expressed_mask_padded = pad_sequence(expressed_mask_list, batch_first=True, padding_value=False)

        # Create attention mask (True for pad tokens, False for real tokens)
        # This matches TransformerEncoder convention where True means ignored
        attention_mask = torch.zeros(taxa_ids_padded.shape, dtype=torch.bool)
        for i, seq in enumerate(taxa_ids_list):
            attention_mask[i, len(seq):] = True

        # Extract batch_ids
        batch_ids = None
        if 'batch_id' in batch[0]:
            batch_ids = torch.tensor([s['batch_id'] for s in batch]).long()

        # apply normalization directly to the original counts
        counts_norm = apply_normalization(self.norm_strategy, counts_padded.numpy(), num_bins=self.num_bins)

        # convert to tensors
        batched = {
            'taxa_ids': taxa_ids_padded,
            'normalized_counts': torch.from_numpy(counts_norm).float(),
            'original_counts': counts_padded,
            'expressed_mask': expressed_mask_padded,
            'attention_mask': attention_mask,
            'depth': depths_original,
            'original_depth': depths_original
        }

        if batch_ids is not None:
            batched['batch_ids'] = batch_ids.long()

        # Add metadata fields if present
        metadata_keys = [k for k in batch[0].keys()
                        if k not in ['taxa_ids', 'original_counts', 'expressed_mask',
                                     'original_depth', 'batch_id']]
        for key in metadata_keys:
            # Stack metadata values
            metadata_values = [s[key] for s in batch]
            # Convert to appropriate tensor type
            if isinstance(metadata_values[0], (int, np.integer)):
                batched[key] = torch.tensor(metadata_values, dtype=torch.long)
            elif isinstance(metadata_values[0], (float, np.floating)):
                batched[key] = torch.tensor(metadata_values, dtype=torch.float)
            else:
                # Keep as list for string or other types
                batched[key] = metadata_values

        return batched


## TODO: Are these functions fast?? Shouldn't this be done in torch?
def apply_normalization(norm_strategy,
                        counts: np.ndarray,
                        num_bins: int = 15) -> np.ndarray:
    """
    Apply normalization strategy to counts.
    :param counts: Count matrix (batch_size, n_taxa).
    :param num_bins: Number of quantile bins used when norm_strategy is 'binning'.
    :return: Normalized count matrix (batch_size, n_taxa).
    """
    if norm_strategy == 'clr':
        # Vectorized CLR to avoid skbio overhead if possible, but keep closure
        counts_pc = counts + 1e-8
        # closure makes it sum to 1
        counts_closed = counts_pc / counts_pc.sum(axis=-1, keepdims=True)
        # log-ratio
        log_counts = np.log(counts_closed)
        clr_counts = log_counts - log_counts.mean(axis=-1, keepdims=True)
        return clr_counts.astype(np.float32)
    elif norm_strategy == 'none':
        return counts.astype(np.float32)
    elif norm_strategy == 'rel_abundance':
        counts_pc = counts + 1e-8
        counts_closed = closure(counts_pc)
        return counts_closed.astype(np.float32)
    elif norm_strategy == 'log_rel_abundance':
        counts_pc = counts + 1e-8
        counts_closed = closure(counts_pc)
        log_rel_abundance = np.log(counts_closed)
        return log_rel_abundance.astype(np.float32)
    elif norm_strategy == 'log_counts':
        counts_pc = counts + 1e-8
        log_counts = np.log(counts_pc)
        return log_counts.astype(np.float32)
    elif norm_strategy == 'binning':
        # bin the counts into N quantile bins, but all zeros go in bin 0
        N = num_bins
        nz = counts > 0
        # handle all-zero samples to avoid nanquantile errors
        any_nz = nz.any(axis=1)
        bins = np.zeros_like(counts, dtype=np.int32)

        if any_nz.any():
            x = np.where(nz[any_nz], counts[any_nz], np.nan).astype(np.float32)
            # quantiles per sample (axis 1) across taxa
            cut = np.nanquantile(x, np.linspace(0, 1, N + 1), axis=1).T[:, 1:-1]  # (B_nz, N-1)
            # handle cases where quantiles are all same (e.g. mostly zeros/ones)
            bins[any_nz] = (counts[any_nz, ..., None] >= cut[:, None, :]).sum(axis=-1).astype(np.int32)

        bins[nz] += 1  # reserve 0 for absence -> bins 1..N for nonzero
        return bins.astype(np.float32)
    elif norm_strategy == 'binary':
        return (counts > 0).astype(np.float32)
    elif norm_strategy == 'arcsine':
        counts_pc = counts + 1e-8
        counts_closed = closure(counts_pc)
        arcsine_transformed = np.arcsin(np.sqrt(counts_closed))
        return arcsine_transformed.astype(np.float32)
    else:
        raise ValueError(f"Unknown normalization strategy: {norm_strategy}")
