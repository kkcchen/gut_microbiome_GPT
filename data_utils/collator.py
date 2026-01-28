"""
Collate function with perturbation
"""
import torch
import numpy as np
from typing import List, Dict, Tuple
from skbio.stats.composition import closure, clr


class MicrobiomeCollator:
    """
    Collator that performs:
    1. Vectorized perturbation (upsample/downsample) on full count vectors
    2. Batch into tensors
    
    This order ensures perturbation affects which taxa are selected as top-k.
    """
    
    def __init__(
        self,
        max_seq_len: int = 200,
        downsample_ratio_range: Tuple[float, float] = (0.5, 0.9),
        upsample_ratio_range: Tuple[float, float] = (1.1, 2.0),
        perturbation_ratio: float = 0.6,
        perturbation_distribution: str = 'zinb',
        perturbation_scale: float = 0.55,
        norm_strategy: str = 'clr'
    ):
        """
        Initialize collator with perturbation and selection parameters.
        
        :param max_seq_len: Number of taxa to select after perturbation.
        :param downsample_ratio_range: Range for downsampling multiplier.
        :param upsample_ratio_range: Range for upsampling multiplier.
        :param perturbation_prob: Probability of applying perturbation per sample.
        :param perturbation_distribution: 'zinb' or 'multinomial'
        """
        self.max_seq_len = max_seq_len
        self.downsample_range = downsample_ratio_range
        self.upsample_range = upsample_ratio_range
        self.r = perturbation_ratio
        self.perturbation_scale = perturbation_scale
        self.perturbation_distribution = perturbation_distribution
        self.norm_strategy = norm_strategy
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate with perturbation.
        
        Process:
        1. Stack all full count vectors (batch_size, max_seq_len)
        2. Apply perturbation to vectors, upsample or downsample
        3. Convert to tensors with fixed length (batch_size, max_seq_len)
        
        Output:
        {
            'taxa_ids': (batch_size, max_seq_len) - selected taxa indices
            'perturbed_counts': (batch_size, max_seq_len) - perturbed selected counts
            'original_counts': (batch_size, max_seq_len) - original selected counts
            'expressed_mask': (batch_size, max_seq_len) - mask for expressed taxa
            'depth': (batch_size,) - total perturbed depth
            'original_depth': (batch_size,) - total original depth
            'batch_ids': (batch_size,) - optional batch labels
            'metadata_fields': varies - optional metadata
        }
        
        :param batch: List of sample dictionaries from dataset.
        :return: Batched tensors.
        """
        batch_size = len(batch)
        
        # Extract data from batch
        taxa_ids_full = np.stack([s['taxa_ids'] for s in batch])  # (B, max_seq_len)
        counts_full = np.stack([s['original_counts'] for s in batch])  # (B, max_seq_len)
        expressed_mask_full = np.stack([s['expressed_mask'] for s in batch])  # (B, max_seq_len)
        depths_original = np.array([s['original_depth'] for s in batch])  # (B,)
        batch_ids = np.array([s['batch_id'] for s in batch]) if 'batch_id' in batch[0] else None

        # perturbation on full count vectors
        counts_perturbed, depths_perturbed = self._perturb_batch(counts_full)

        # apply normalization
        counts_perturbed_norm = self._apply_normalization(counts_perturbed) 
        
        # convert to tensors
        batched = {
            'taxa_ids': torch.from_numpy(taxa_ids_full).long(),
            'perturbed_counts': torch.from_numpy(counts_perturbed_norm).float(),
            'original_counts': torch.from_numpy(counts_full).float(),
            'expressed_mask': torch.from_numpy(expressed_mask_full).bool(),
            'depth': torch.from_numpy(depths_perturbed).float(),
            'original_depth': torch.from_numpy(depths_original).float()
            
        }
        
        if batch_ids is not None:
            batch_ids = np.array([s['batch_id'] for s in batch])
            batched['batch_ids'] = torch.from_numpy(batch_ids).long()
        
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
    
    def _apply_normalization(self,
                             counts: np.ndarray) -> np.ndarray:
        """
        Apply normalization strategy to perturbed counts.
        :param counts: Perturbed count matrix (batch_size, n_taxa).
        :return: Normalized count matrix (batch_size, n_taxa).
        """
        if self.norm_strategy == 'clr':
            # Add pseudocount to avoid log(0)
            counts_pc = counts + 1.0
            counts_closed = closure(counts_pc)
            clr_counts = clr(counts_closed)
            return clr_counts.astype(np.float32)
        elif self.norm_strategy == 'none':
            return counts.astype(np.float32)
        else:
            raise ValueError(f"Unknown normalization strategy: {self.norm_strategy}")

    def _perturb_batch(
        self,
        counts: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply vectorized perturbation to entire batch.
        
        :param counts: Full count matrix (batch_size, n_total_taxa).
        :param batch_size: Number of samples in batch.
        :return: Tuple of (perturbed_counts, perturbed_depths).
        """
        # determine perturbation approach
        if self.perturbation_distribution == 'multinomial':
            resample_fn = self._multinomial_resample
        elif self.perturbation_distribution == 'zinb':
            resample_fn = self._zinb_resample
        else:
            raise ValueError(f"Unknown perturbation distribution: {self.perturbation_distribution}")
        # fast perturbation
        return resample_fn(counts)
    
    def _multinomial_resample(
        self,
        counts: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorized multinomial resampling for entire batch.
        
        :param counts: Count matrix (batch_size, n_total_taxa).
        :return: Tuple of (perturbed_counts, perturbed_depths).
        """
        # not implemented
        raise NotImplementedError("Multinomial resampling not implemented yet.")
    
    def _zinb_resample(
        self,
        counts: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        From scPRINT: ZINB perturbation, this is viewed as a downsampling process of original data.
        x_hat_i = max((x_i - p_i) * pi_i, 0)
        where
        x_hat_i: perturbed sample
        x_i: original sample
        p_i: drawn from Poisson(x_i x r x self.perturbation_scale)
        pi_i: I(u >= r x self.perturbation_scale), binary mask indicating non dropout taxa
        u_i: drawn from Uniform(0, 1), 
        
        :param counts: Count matrix (batch_size, n_total_taxa).
        :return: Tuple of (perturbed_counts, perturbed_depths).
        """
        batch_size, n_taxa = counts.shape
        # draw u
        u = np.random.uniform(0, 1, size=(batch_size, n_taxa)) # B, N
        # compute pi
        pi = (u >= self.r * self.perturbation_scale).astype(np.float32)  # B, N
        # draw p
        lambda_param = counts * self.r * self.perturbation_scale
        p = np.random.poisson(lam=lambda_param)  # B, N
        # compute x_hat
        perturbed_counts = np.maximum((counts - p) * pi, 0)  # B, N
        perturbed_depths = perturbed_counts.sum(axis=1)  # B,
        return perturbed_counts.astype(np.float32), perturbed_depths.astype(np.float32)
