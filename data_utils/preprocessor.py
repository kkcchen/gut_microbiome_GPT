import numpy as np
from typing import Dict, Optional, Union, List
from pandas import DataFrame as df
import pandas as pd
import anndata as ad
from skbio.stats.composition import clr, closure, multi_replace
from trainers import logger

class Preprocessor:
    """
    currently just bins, and filters by prev and abundance. could do other preprocessing steps in the future. 
    """

    def __init__(
        self,
        binning: Optional[int] = None,
        keep_top_k: Optional[int] = 512,
    ):
        r"""
        Set up the preprocessor, use the args to config the workflow steps.

        Args:
        binning (:class:`int`, optional):
            Whether to bin the data into discrete values of number of bins provided.
        """
        self.binning = binning
        self.keep_top_k = keep_top_k

    def bin_from_np(self, unprocessed_data: np.ndarray, taxa_ids) -> Dict:
        """
        Process the unprocessed data from a numpy array. Apply binning.

        Args:
        unprocessed_data (:class:`np.ndarray`):
            The unprocessed data. size (num_samples, num_taxa)


        Returns:
        :class:`np.ndarray`:
            The preprocessed data.
        :class:`np.ndarray`:
            The bin edges of the data.
        """
        assert len(taxa_ids) == unprocessed_data.shape[1], "The number of taxa IDs must match the number of columns in the data."
        
        if not isinstance(unprocessed_data, np.ndarray):
            raise ValueError("The unprocessed data must be a numpy array.")
        
        # filtering, keep top k
        if self.keep_top_k is not None:
            top_k_indices = np.argsort(unprocessed_data, axis=1)[:, -self.keep_top_k:]
            mask = np.zeros_like(unprocessed_data, dtype=bool)
            np.put_along_axis(mask, top_k_indices, True, axis=1)
            removed_entries = np.sum((~mask) & (unprocessed_data > 0))
            logger.info(f"Number of taxa removed in total: {removed_entries} for {unprocessed_data.shape[0]} samples.")
            unprocessed_data = np.where(mask, unprocessed_data, 0)
        
        # binning
        if not self.binning:
            raise ValueError("Binning is not enabled, should this be the case?")
        
        n_bins = self.binning  # NOTE: the first bin is always a special for zero
        binned_rows = []
        bin_edges = []

        # Iterate over each row 
        for row in unprocessed_data:
            if row.max() == 0:
                raise ValueError(
                    "The data has all zero values, please check the data."
                )
                binned_rows.append(np.zeros_like(row, dtype=np.int64))
                bin_edges.append(np.array([0] * n_bins))
                continue

            non_zero_ids = row.nonzero()
            non_zero_row = row[non_zero_ids]
            bins = np.quantile(non_zero_row, np.linspace(0, 1, n_bins - 1))
            # bins = np.sort(np.unique(bins))
            # NOTE: comment this line for now, since this will make the each category
            # has different relative meaning across datasets
            non_zero_digits = _digitize(non_zero_row, bins) + 1

            # print(non_zero_digits, row, bins)
            assert non_zero_digits.min() >= 1
            assert non_zero_digits.max() <= n_bins - 1
            binned_row = np.zeros_like(row, dtype=np.int64)
            binned_row[non_zero_ids] = non_zero_digits
            binned_rows.append(binned_row)
            bin_edges.append(np.concatenate([[0], bins]))
                
        return np.stack(binned_rows), np.stack(bin_edges)

    def clr_from_np(self, unprocessed_data: np.ndarray, taxa_ids) -> Dict:
        """
        Process the unprocessed data from a numpy array. Apply CLR transform only

        Args:
        unprocessed_data (:class:`np.ndarray`):
            The unprocessed data. size (num_samples, num_taxa)


        Returns:
        :class:`np.ndarray`:
            The preprocessed data.
        :class:`np.ndarray`:
            The bin edges of the data.
        """
        assert len(taxa_ids) == unprocessed_data.shape[1], "The number of taxa IDs must match the number of columns in the data."
        
        print("Not doing binning, only applying CLR transform to data!")
        clr_data = []

        # Iterate over each row 
        for row in unprocessed_data:
            if row.max() == 0:
                raise ValueError(
                    "The data has all zero values, please check the data."
                )
            
            # Get non-zero indices and values
            non_zero_mask = row > 0
            non_zero_values = row[non_zero_mask]
            
            # Apply CLR to non-zero values
            log_non_zero = np.log(non_zero_values)
            geometric_mean_log = np.mean(log_non_zero)
            clr_non_zero = log_non_zero - geometric_mean_log

            # the above produces negative values, as an experiment shift rows so no negative
            epsilon = 1e-6
            row_min, row_max = clr_non_zero.min(), clr_non_zero.max()
            if row_max > row_min:
                clr_non_zero_scaled = ((clr_non_zero - row_min) / (row_max - row_min)) * (1 - epsilon) + epsilon
            else:
                clr_non_zero_scaled = np.full_like(clr_non_zero, fill_value=epsilon)
            
            # Create output array with zeros preserved
            clr_row = np.zeros_like(row, dtype=np.float64)
            clr_row[non_zero_mask] = clr_non_zero_scaled
            
            clr_data.append(clr_row)
        
        clr_array = np.stack(clr_data)
                
        return clr_array


    def clr_transform(self, unprocessed_data: np.ndarray, taxa_ids) -> np.ndarray:
        """
        Perform centered log-ratio transformation on the data.

        Args:
        data (:class:`np.ndarray`):
            The data to be transformed. size (num_samples, num_taxa)
        """
        if not isinstance(unprocessed_data, np.ndarray):
            raise ValueError("The data must be a numpy array.")
        
        if np.any(unprocessed_data < 0):
            raise ValueError("The data must be non-negative.")
        
        gm = np.exp(np.mean(np.log(unprocessed_data), axis=1))
        clr_data = np.log(unprocessed_data / gm[:, None])
        return clr_data
    
def compute_prevalence_abundance(
    data, 
    var_names=None
):
    """
    Compute prevalence (fraction of samples > 0) and abundance (mean value)
    for features in either an AnnData object or a NumPy array.

    Parameters
    ----------
    data : AnnData | np.ndarray
        Input data. If AnnData, uses data.X. If ndarray, uses it directly.
        Shape should be (n_samples, n_features).
    var_names : list[str] | None, optional
        Feature names. Required if data is a NumPy array.

    Returns
    -------
    prevalence : pd.Series
        Fraction of samples with nonzero values for each feature.
    abundance : pd.Series
        Mean abundance of each feature across samples.
    """
    if isinstance(data, ad.AnnData):
        X = data.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        var_names = data.var_names
    elif isinstance(data, np.ndarray):
        X = data
        if var_names is None:
            raise ValueError("var_names must be provided when data is a NumPy array.")
    else:
        raise TypeError("Input must be an AnnData object or a NumPy array.")

    prevalence = np.mean(X > 0, axis=0)
    abundance = np.mean(X, axis=0)

    return (
        pd.Series(prevalence, index=var_names),
        pd.Series(abundance, index=var_names),
    )

def preprocess_clr_matrix(X):
    if hasattr(X, "toarray"):
        X = X.toarray()
    X_replaced = multi_replace(X)
    X_closed = closure(X_replaced)
    return clr(X_closed)



def _digitize(x: np.ndarray, bins: np.ndarray, side="both") -> np.ndarray:
    """
    Digitize the data into bins. This method spreads data uniformly when bins
    have same values.

    Args:

    x (:class:`np.ndarray`):
        The data to digitize.
    bins (:class:`np.ndarray`):
        The bins to use for digitization, in increasing order.
    side (:class:`str`, optional):
        The side to use for digitization. If "one", the left side is used. If
        "both", the left and right side are used. Default to "both".

    Returns:

    :class:`np.ndarray`:
        The digitized data.
    """
    assert x.ndim == 1 and bins.ndim == 1

    if side not in ("both", "one"):
        raise ValueError(f"side must be 'both' or 'one', got {side}")

    left_digits = np.digitize(x, bins)
    if side == "one":
        return left_digits

    right_digits = np.digitize(x, bins, right=True)
    right_digits[right_digits == len(bins)] = len(bins) - 1

    rands = np.random.rand(*x.shape)  # uniform random numbers
    digits = np.where(rands < 0.5, left_digits, right_digits)

    # make sure the digits are in the range of [0, len(bins)], because left binning can produce len(bins) in the list
    digits[digits == len(bins)] = len(bins) - 1

    return digits