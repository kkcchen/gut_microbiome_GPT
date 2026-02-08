"""
Collate function with perturbation
"""
import torch
import numpy as np
from typing import List, Dict, Tuple
from skbio.stats.composition import closure, clr

# TODO: How do we want to handle random number generation? Generator passed in __init__, created in __init__, or created in each function that needs it.
#       For now, initialize in __init__
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
        norm_strategy: str = 'clr',
        do_contrastive: bool = False,
        eval_mode: bool = False,
    ):
        """
        Initialize collator with perturbation and selection parameters.
        
        :param max_seq_len: Number of taxa to select after perturbation.
        :param downsample_ratio_range: Range for downsampling multiplier.
        :param upsample_ratio_range: Range for upsampling multiplier.
        :param perturbation_prob: Probability of applying perturbation per sample.
        :param perturbation_distribution: 'zinb' | 'multinomial' | 'binomial'
        """
        self.max_seq_len = max_seq_len
        self.downsample_range = downsample_ratio_range
        self.upsample_range = upsample_ratio_range
        self.r = perturbation_ratio
        self.perturbation_scale = perturbation_scale
        self.perturbation_distribution = perturbation_distribution
        self.norm_strategy = norm_strategy
        self.do_contrastive = do_contrastive
        self.eval_mode = eval_mode
        self.rng = np.random.default_rng()
    
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
        if self.eval_mode:
            counts_perturbed = counts_full.copy()
            depths_perturbed = depths_original.copy()
        else:
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
        
        if self.do_contrastive:
            # create a second perturbed view for contrastive learning
            counts_perturbed_2, depths_perturbed_2 = self._perturb_batch(counts_full)
            counts_perturbed_norm_2 = self._apply_normalization(counts_perturbed_2)
            batched['perturbed_counts_2'] = torch.from_numpy(counts_perturbed_norm_2).float()
            batched['depth_2'] = torch.from_numpy(depths_perturbed_2).float()
        
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
    
    ## TODO: Are these functions fast?? Shouldn't this be done in torch?
    def _apply_normalization(self,
                             counts: np.ndarray) -> np.ndarray:
        """
        Apply normalization strategy to perturbed counts.
        :param counts: Perturbed count matrix (batch_size, n_taxa).
        :return: Normalized count matrix (batch_size, n_taxa).
        """
        if self.norm_strategy == 'clr':
            # Add pseudocount to avoid log(0)
            counts_pc = counts + 1e-8 # TODO pseudo-count could be a parameter or in config
            counts_closed = closure(counts_pc)
            clr_counts = clr(counts_closed)
            return clr_counts.astype(np.float32)
        elif self.norm_strategy == 'none':
            return counts.astype(np.float32)
        elif self.norm_strategy == 'rel_abundance':
            counts_pc = counts + 1e-8
            counts_closed = closure(counts_pc)
            return counts_closed.astype(np.float32)
        elif self.norm_strategy == 'log_rel_abundance':
            counts_pc = counts + 1e-8
            counts_closed = closure(counts_pc)
            log_rel_abundance = np.log(counts_closed)
            return log_rel_abundance.astype(np.float32)
        elif self.norm_strategy == 'log_counts':
            counts_pc = counts + 1e-8
            log_counts = np.log(counts_pc)
            return log_counts.astype(np.float32)
        elif self.norm_strategy == 'binning': # TODO: Test this
            # bin the counts into N quantile bins, but all zeros go in bin 0
            N = 50 # TODO: make N a parameter
            bins = np.zeros_like(counts, dtype=int)
            nz = counts > 0
            x = np.where(nz, counts, np.nan).astype(np.float32)
            cut = np.nanquantile(x, np.linspace(0, 1, N + 1), axis=1).transpose(1, 0)[:, 1:-1]  # (B, N-1)
            bins = (counts[..., None] >= cut[:, None, :]).sum(axis=-1).astype(np.int32)
            bins[~nz] = 0
            bins[nz] += 1  # reserve 0 for absence -> bins 1..N for nonzero
            # edges = np.quantile(counts[nz], np.linspace(0, 1, N + 1))
            # bins[nz] = np.digitize(counts[nz],edges[1:-1]) + 1
            return bins.astype(np.float32)
        elif self.norm_strategy == 'arcsine':
            counts_pc = counts + 1e-8
            counts_closed = closure(counts_pc)
            arcsine_transformed = np.arcsin(np.sqrt(counts_closed))
            return arcsine_transformed.astype(np.float32) 
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
        elif self.perturbation_distribution == 'binomial':
            resample_fn = self._binomial_resample
        else:
            raise ValueError(f"Unknown perturbation distribution: {self.perturbation_distribution}")
        # fast perturbation
        return resample_fn(counts)
    
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
    
    def _multinomial_resample(
        self,
        counts: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorized multinomial resampling for entire batch.
        
        :param counts: Count matrix (batch_size, n_total_taxa).
        :return: Tuple of (perturbed_counts, perturbed_depths).
        """
        B, L = counts.shape
        total_counts = counts.sum(axis=1).astype(np.int64)  # (B,)
        low, high = self.downsample_range
        ratios = self.rng.uniform(low, high, size=B)
        target_totals = np.floor(total_counts * ratios).astype(np.int64)  # (B,)
        ps = counts / total_counts[:, None]  # (B, L)
        perturbed_counts = self.rng.multinomial(n=target_totals, pvals=ps).astype(counts.dtype, copy=False)
        return perturbed_counts, target_totals
    
    
    ### TODO torch instead of numpy?
    def _binomial_resample(
        self,
        counts: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Binomial thinning, as a more efficient alternative to multinomial resampling.
        
        :param counts: Count matrix (batch_size, n_total_taxa).
        :return: Tuple of (perturbed_counts, perturbed_depths).
        """
        B, L = counts.shape

        low, high = self.downsample_range
        ratios = self.rng.uniform(low, high, size=B).astype(np.float64)  # (B,)

        # Broadcast ratios across taxa: (B, 1) -> (B, L)
        p = ratios[:, None]

        # Vectorized binomial draws across the full matrix
        perturbed_counts = self.rng.binomial(n=counts, p=p).astype(counts.dtype, copy=False)

        # Realized depths after thinning (random)
        perturbed_totals = perturbed_counts.sum(axis=1).astype(np.int64)

        return perturbed_counts, perturbed_totals
