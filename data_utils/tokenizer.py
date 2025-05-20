import numpy as np
from typing import Dict, Optional, Union, List, Tuple
from pandas import DataFrame as df
import torch

class MicrobiomeVocab():
    def __init__(
        self,
        data: df,
        class_token: str = "<cls>",
        mask_token: str = "<mask>",
        pad_token: str = "<pad>",
        pad_value: int = 0,
        mask_value: int = -1,
    ):
        """
        Initialize the vocabulary with taxa and special tokens.
        Args:
            taxa (df): A DataFrame containing taxa names. columns are taxa names.
            data frame should be in this format:
                | sample | taxa0 | ... | taxaN |
                |--------|-------|-----|-------|
                |   s1   |   1   | ... |   0   |
                |   s2   |   0   | ... |   1   |
            class_token (str): The token representing the class.
            mask_token (str): The token representing the mask.
        """
        # check the data follows the format
        if not isinstance(data, df):
            raise ValueError("Data should be a pandas DataFrame.")
        if not all(isinstance(col, str) for col in data.columns[1:]):
            raise ValueError("All taxa names should be strings.")
        if not all(isinstance(x, (int, np.integer)) for x in data.iloc[:, 1:].values.flatten()):
            raise ValueError("All taxa values should be numeric.")
        

        self.data = data
        self.itos = data.columns.tolist()[1:] + [class_token, mask_token, pad_token]
        # assert there are no duplicates in the taxa names
        if len(self.itos) != len(set(self.itos)):
            raise ValueError("Duplicate taxa names found in the DataFrame.")
        
        self.stoi = {token: idx for idx, token in enumerate(self.itos)}
        self.class_token = class_token
        self.mask_token = mask_token
        self.pad_token = pad_token
        self.pad_value = pad_value
        self.mask_value = mask_value

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: str):
        return self.stoi.get(item, None)
    
    def lookup_indices(self, taxa_names: List[str]) -> List[int]:
        """
        Lookup the indices of the taxa_ids in the vocabulary.
        Args:
            taxa_names (List[str]): A list of taxa names.
        Returns:
            List[int]: A list of indices of the taxa_ids in the vocabulary.
        """
        return [self.stoi.get(taxa_name, None) for taxa_name in taxa_names]


class Tokenizer:
    def __init__(self, vocab: MicrobiomeVocab):
        self.vocab = vocab

    def tokenize_batch(
        self,
        return_pt: bool = True,
        prepend_cls: bool = True,
        include_zero_count: bool = False,
    ) -> List[Tuple[Union[torch.Tensor, np.ndarray]]]:
        """
        Tokenize a batch of data. Returns a list of tuple (array_like taxa_id, array_like values).

        Args:
            data (array-like): A batch of data, with shape (batch_size, n_features).
                n_features equals the number of all taxa.
            return_pt (bool): Whether to return torch tensors of gene_ids and counts,
                default to True.

        Returns:
            list: A list of tuple (gene_id, count) of non zero gene expressions.
        """
        tokenized_data = []
        for i in range(len(self.vocab)):
            row = self.vocab.data.iloc[i, 1:].to_numpy()
            if include_zero_count:
                values = row
                taxa_names = self.vocab.data.columns[1:]
            else:
                idx = np.nonzero(row)[0]
                values = row[idx]
                taxa_names = self.vocab.data.columns[1:].to_numpy()[idx]

            if prepend_cls:
                taxa_names = np.insert(taxa_names, 0, self.vocab.class_token)
                values = np.insert(values, 0, self.vocab.pad_value)

            taxa_id = np.array(self.vocab.lookup_indices(taxa_names), dtype=np.int64)
            values = np.array(values, dtype=np.int64)

            if return_pt:
                taxa_id = torch.from_numpy(taxa_id).long()
                values = torch.from_numpy(values).float()

            tokenized_data.append((taxa_id, values))
        return tokenized_data
    
    def pad_batch(
        self,
        batch: List[Tuple],
        max_len: int,
        cls_prepended: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Pad a batch of data. Returns a list of Dict[taxa_id, count].

        Args:
            batch (list): A list of tuple (array_like taxa_id, array_like values).
            max_len (int): The maximum length of the batch.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of taxa_id and values.
        """
        max_ori_len = max(len(batch[i][0]) for i in range(len(batch)))
        max_len = min(max_ori_len, max_len)

        pad_id = self.vocab[self.vocab.pad_token]
        # if vocab_mod is not None:
        #     mod_pad_id = vocab_mod[pad_token]
        taxa_ids_list = []
        values_list = []
        mod_types_list = []

        # for i in range(len(batch)):
        #     gene_ids, values, mod_types = batch[i]
        for taxa_ids, values in batch:
            if len(taxa_ids) > max_len:
                # take most abundant taxa
                if not cls_prepended:
                    # idx = np.random.choice(len(gene_ids), max_len, replace=False)
                    idx = np.argsort(values)[-max_len:]
                else:
                    # idx = np.random.choice(len(gene_ids) - 1, max_len - 1, replace=False)
                    idx = np.argsort(values[1:])[-(max_len - 1):]
                    idx = idx + 1
                    # sort the idx to get the same order as the original gene_ids
                    idx.sort()
                    idx = np.insert(idx, 0, 0)
                taxa_ids = taxa_ids[idx]
                values = values[idx]
                # if mod_types is not None:
                #     mod_types = mod_types[idx]
            elif len(taxa_ids) < max_len:
                taxa_ids = torch.cat(
                    [
                        taxa_ids,
                        torch.full(
                            (max_len - len(taxa_ids),), pad_id, dtype=taxa_ids.dtype
                        ),
                    ]
                )
                values = torch.cat(
                    [
                        values,
                        torch.full((max_len - len(values),), self.vocab.pad_value, dtype=values.dtype),
                    ]
                )
                # if mod_types is not None:
                #     mod_types = torch.cat(
                #         [
                #             mod_types,
                #             torch.full(
                #                 (max_len - len(mod_types),),
                #                 mod_pad_id,
                #                 dtype=mod_types.dtype,
                #             ),
                #         ]
                #     )

            taxa_ids_list.append(taxa_ids)
            values_list.append(values)
            # if mod_types is not None:
            #     mod_types_list.append(mod_types)

        batch_padded = {
            "taxa": torch.stack(taxa_ids_list, dim=0),
            "values": torch.stack(values_list, dim=0),
        }
        # if mod_types is not None:
        #     batch_padded["mod_types"] = torch.stack(mod_types_list, dim=0)
        return batch_padded

    def tokenize_and_pad_batch(
        self,
        max_len: int,
        prepend_cls: bool = True,
        include_zero_count: bool = False,
        return_pt: bool = True,
        # mod_type: np.ndarray = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize and pad a batch of data. Returns a list of tuple (gene_id, count).
        """
        # if mod_type is not None:
        #     cls_id_mod_type = vocab_mod[cls_token]
        tokenized_data = self.tokenize_batch(
            return_pt=return_pt,
            prepend_cls=prepend_cls,
            include_zero_count=include_zero_count,
        )

        batch_padded = self.pad_batch(
            tokenized_data,
            max_len=max_len,
            cls_prepended=prepend_cls,
        )
        return batch_padded
    
    def random_mask_value(
    self,
    values: Union[torch.Tensor, np.ndarray],
    mask_ratio: float = 0.15,
    ) -> torch.Tensor:
        """
        Randomly mask a batch of data.

        Args:
            values (array-like):
                A batch of tokenized data, with shape (batch_size, n_features).
            mask_ratio (float): The ratio of genes to mask, default to 0.15.=
        Returns:
            torch.Tensor: A tensor of masked data.
        """
        # pad_value must be kept unchanged
        if isinstance(values, torch.Tensor):
            # it is crutial to clone the tensor, otherwise it changes the original tensor
            values = values.clone().detach().numpy()
        else:
            values = values.copy()

        for row in values:
            non_padding_idx = np.where(row != self.vocab.pad_value)[0]
            n_mask = int(len(non_padding_idx) * mask_ratio)
            mask_idx = np.random.choice(non_padding_idx, n_mask, replace=False)
            row[mask_idx] = self.vocab.mask_value
        return torch.from_numpy(values).float()