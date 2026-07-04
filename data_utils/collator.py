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
        finetune_mode: bool = False,
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
        self.finetune_mode = finetune_mode
        self.rng = np.random.default_rng()
    
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

        # Handle perturbation on padded tensors
        counts_padded_np = counts_padded.numpy()
        if self.eval_mode or self.finetune_mode:
            counts_perturbed_np = counts_padded_np.copy()
            depths_perturbed = depths_original.clone()
        else:
            counts_perturbed_np, depths_perturbed_np = self._perturb_batch(counts_padded_np)
            depths_perturbed = torch.from_numpy(depths_perturbed_np).float()
            
        # apply normalization
        counts_perturbed_norm = apply_normalization(self.norm_strategy, counts_perturbed_np) 
        
        # convert to tensors
        batched = {
            'taxa_ids': taxa_ids_padded,
            'perturbed_counts': torch.from_numpy(counts_perturbed_norm).float(),
            'original_counts': counts_padded,
            'expressed_mask': expressed_mask_padded,
            'attention_mask': attention_mask,
            'depth': depths_perturbed,
            'original_depth': depths_original
        }
        
        if self.do_contrastive:
            # create a second perturbed view for contrastive learning
            counts_perturbed_2_np, depths_perturbed_2_np = self._perturb_batch(counts_padded_np)
            counts_perturbed_norm_2 = apply_normalization(self.norm_strategy, counts_perturbed_2_np)
            batched['perturbed_counts_2'] = torch.from_numpy(counts_perturbed_norm_2).float()
            batched['depth_2'] = torch.from_numpy(depths_perturbed_2_np).float()
        
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
        """
        batch_size, n_taxa = counts.shape
        # draw u and p vectorized
        u = self.rng.uniform(0, 1, size=(batch_size, n_taxa)) # B, N
        pi = (u >= self.r * self.perturbation_scale).astype(np.float32)  # B, N
        
        lambda_param = counts * (self.r * self.perturbation_scale)
        p = self.rng.poisson(lam=lambda_param)  # B, N
        
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
        # counts must be integer for binomial n parameter
        perturbed_counts = self.rng.binomial(n=counts.astype(np.int64), p=p).astype(counts.dtype, copy=False)

        # Realized depths after thinning (random)
        perturbed_totals = perturbed_counts.sum(axis=1).astype(np.int64)

        return perturbed_counts, perturbed_totals

## TODO: Are these functions fast?? Shouldn't this be done in torch?
def apply_normalization(norm_strategy,
                        counts: np.ndarray) -> np.ndarray:
    """
    Apply normalization strategy to perturbed counts.
    :param counts: Perturbed count matrix (batch_size, n_taxa).
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
        N = 15 # TODO: make N a parameter
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