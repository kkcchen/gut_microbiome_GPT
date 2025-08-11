import numpy as np
from typing import Dict, Optional, Union, List
from pandas import DataFrame as df

class Preprocessor:
    """
    currently just bins. could do other preprocessing steps in the future. 
    """

    def __init__(
        self,
        binning: Optional[int] = None,
    ):
        r"""
        Set up the preprocessor, use the args to config the workflow steps.

        Args:
        binning (:class:`int`, optional):
            Whether to bin the data into discrete values of number of bins provided.
        """
        self.binning = binning

    def process_from_np(self, unprocessed_data: np.ndarray, taxa_ids) -> Dict:
        """
        Process the unprocessed data from a numpy array.

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
        
        if not self.binning:
            raise ValueError("Binning is not enabled, should this be the case?")
            
        n_bins = self.binning  # NOTE: the first bin is always a spectial for zero
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