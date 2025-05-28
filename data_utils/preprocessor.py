import numpy as np
from typing import Dict, Optional, Union, List
from pandas import DataFrame as df

class BatchVocab():
    """
    A class to represent the vocabulary of batches in the dataset.
    """

    def __init__(self, vocab: np.ndarray):
        """
        Initialize the vocabulary with taxa and special tokens.

        Args:
            vocab (np.ndarray): A numpy array containing batch names. 
                The first column should be the sample names, and the rest are batch names
        """
        # get the unique batch names
        self.vocab = vocab
        self.itos = vocab[:].tolist()
        self.stoi = {token: idx for idx, token in enumerate(self.itos)}

        # make sure there are no duplicates in the batch names
        if len(self.itos) != len(set(self.itos)):
            raise ValueError("Duplicate batch names found in the DataFrame.")
        
    def __getitem__(self, item: str):
        return self.stoi.get(item, None)

class Preprocessor:
    """
    currently just bins. could do other preprocessing steps in the future. 
    """

    def __init__(
        self,
        filter_gene_by_counts: Union[int, bool] = False,
        binning: Optional[int] = None,
    ):
        r"""
        Set up the preprocessor, use the args to config the workflow steps.

        Args:
        filter_gene_by_counts (:class:`int`, optional):
            Whether to filter genes by counts, if :class:`int`, filter genes with counts
        binning (:class:`int`, optional):
            Whether to bin the data into discrete values of number of bins provided.
        """
        self.binning = binning
        self.filter_gene_by_counts = filter_gene_by_counts

    def process_from_df(self, unprocessed_data: df) -> Dict:
        """
        format controls the different input value wrapping, including categorical
        binned style, fixed-sum normalized counts, log1p fixed-sum normalized counts, etc.

        Args:

        unprocessed_data (:class:`df`):
            The :class:`df` object to preprocess. size (num_samples, num_taxa)

        Returns:
        :class:`np.ndarray`:
            The preprocessed data.
        :class:`np.ndarray`:
            The bin edges of the data.
        """
        # delete first column
        unprocessed_data.drop(unprocessed_data.columns[0], axis=1, inplace=True)

        # step 1: filter genes
        # skipped 

        # step 2: filter cells
        # skipped

        # step 3: normalize total
        # skipped

        # step 4: log1p
        # skipped

        # step 5: subset hvg
        # skipped

        # step 6: binning
        if self.binning:
            if not isinstance(self.binning, int):
                raise ValueError(
                    "Binning arg must be an integer, but got {}.".format(self.binning)
                )
            n_bins = self.binning  # NOTE: the first bin is always a spectial for zero
            binned_rows = []
            bin_edges = []

            # Iterate over each row (sample) in the DataFrame, excluding non-numeric columns
            numeric_data = unprocessed_data.select_dtypes(include=[np.number])
            for _, row in numeric_data.iterrows():
                row = row.values  # convert Series to np array for processing
                if row.max() == 0:
                    # raise ValueError(
                    #     "The data has all zero values, please check the data."
                    # )
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
        else:
            raise ValueError("Binning is not enabled, should this be the case?")
                
        # Update the original DataFrame with binned data (only numeric columns)
        numeric_cols = unprocessed_data.select_dtypes(include=[np.number]).columns
        unprocessed_data.loc[:, numeric_cols] = np.stack(binned_rows)
        return np.stack(binned_rows), np.stack(bin_edges)
    

    def process_from_np(self, unprocessed_data: np.ndarray) -> Dict:
        """
        Process the unprocessed data from a numpy array.

        Args:
        unprocessed_data (:class:`np.ndarray`):
            The unprocessed data. size (num_samples, num_taxa, 2), {:,:, 0} is the taxa id, {:,:, 1} is the value


        Returns:
        :class:`np.ndarray`:
            The preprocessed data.
        :class:`np.ndarray`:
            The bin edges of the data.
        """
        if not isinstance(unprocessed_data, np.ndarray):
            raise ValueError("The unprocessed data must be a numpy array.")
        
        if not self.binning:
            raise ValueError("Binning is not enabled, should this be the case?")

            
        n_bins = self.binning  # NOTE: the first bin is always a spectial for zero
        binned_rows = []
        bin_edges = []

        # Iterate over each row 
        for sample in unprocessed_data:
            row = sample[:, 1]
            if row.max() == 0:
                # raise ValueError(
                #     "The data has all zero values, please check the data."
                # )
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
                
        # Update the original DataFrame with binned data (only numeric columns)
        numeric_cols = unprocessed_data.select_dtypes(include=[np.number]).columns
        unprocessed_data.loc[:, numeric_cols] = np.stack(binned_rows)
        return np.stack(binned_rows), np.stack(bin_edges)
        

    def get_batch_labels(self, unprocessed_data: df) -> np.ndarray:
        """
        Get the batch labels of the data.

        Args:
        unprocessed_data (:class:`df`):
            The unprocessed data. There should be a column named "sample" in the data, with experiment_srr format

        Returns:
        :class:`np.ndarray`:
            The batch labels.
        """
        if "sample" not in unprocessed_data.columns:
            raise ValueError(
                "The unprocessed data must have a column named 'sample' to get batch labels."
            )
        sample_srr = unprocessed_data['sample'].str.split('_', n=1, expand=True)

        # make a list with no repeats
        unique_samples = sample_srr[0].unique()

        batch_vocab = BatchVocab(unique_samples)

        # add a column to the unprocessed_data with the batch labels
        unprocessed_data['batch'] = sample_srr[0]

        # convert the batch labels to indices
        unprocessed_data['batch'] = unprocessed_data['batch'].map(batch_vocab.__getitem__)
        # move the batch column to the front
        unprocessed_data = unprocessed_data[['batch'] + [col for col in unprocessed_data.columns if col != 'batch']]
    
        return BatchVocab(unique_samples).itos


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