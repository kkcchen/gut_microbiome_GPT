import numpy as np
from typing import Dict, Optional, Union, List, Tuple
import torch

from data_utils.vocab import MicrobiomeVocab, BatchVocab
import anndata as ad
from trainers import logger

class Tokenizer:
    def __init__(self, vocab: MicrobiomeVocab):
        self.vocab = vocab

    def tokenize_batch(
        self,
        data: ad.AnnData,
        return_pt: bool = True,
        prepend_cls: bool = True,
        include_zero_count: bool = False,
    ) -> List[Tuple[Union[torch.Tensor, np.ndarray]]]:
        """
        Tokenize a batch of data. Returns a list of tuple (array_like taxa_id, array_like values).

        Args:
            data (anndata): A batch of data, with shape (num_samples, n_taxa)
            return_pt (bool): Whether to return torch tensors of gene_ids and counts,
                default to True.

        Returns:
            list: A list of tuple (taxa_id, count) of non zero gene expressions.
        """
        tokenized_data = []
        for sample in data:
            if include_zero_count:
                values = sample.layers["binned_rows"]
                taxa_ids = sample.var["taxa_id"]
            else:
                idx = np.nonzero(sample.layers["binned_rows"])
                if len(idx[0]) == 0:
                    raise ValueError("Sample has a non-zero counts.")
                values = sample.layers["binned_rows"][idx]
                taxa_ids = sample.var["taxa_id"].iloc[idx[1]]

            if prepend_cls:
                taxa_ids = np.insert(taxa_ids, 0, self.vocab.class_index)
                values = np.insert(values, 0, self.vocab.pad_value)

            if return_pt:
                taxa_ids = torch.from_numpy(taxa_ids).long()   
                values = torch.from_numpy(values).float()

            tokenized_data.append((taxa_ids, values))

        logger.info(f"Tokenized sample with {len(tokenized_data)} samples.")
        return tokenized_data

        # if prepend_cls:
        #     cls_idx = self.vocab.class_index
        #     cls_count = self.vocab.pad_value
        #     data_prepend = torch.cat(
        #         [
        #             torch.tensor([cls_idx, cls_count], dtype=torch.float32).reshape(1, 1, 2),
        #             data,
        #         ],
        #         dim=2,
        #     )

        #     return [
        #         (
        #             torch.from_numpy(row[:, 0]),
        #             torch.from_numpy(row[:, 1])
        #         ) for row in data_prepend
        #     ]
        
        # else:
        #     return [
        #         (
        #             torch.from_numpy(row[:, 0]), 
        #             torch.from_numpy(row[:, 1])
        #         ) for row in data
        #     ]

    def pad_batch(
        self,
        batch: List[Tuple],
        max_len: Optional[int] = None,
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
        if max_len is not None:
            max_len = min(max_ori_len, max_len)
        else:
            max_len = max_ori_len

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
                            (max_len - len(taxa_ids),), self.vocab.pad_index, dtype=taxa_ids.dtype
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
            "taxa_ids": torch.stack(taxa_ids_list, dim=0),
            "values": torch.stack(values_list, dim=0),
        }
        # if mod_types is not None:
        #     batch_padded["mod_types"] = torch.stack(mod_types_list, dim=0)
        return batch_padded
    
    def add_labels(
        self,
        data_dict: Dict[str, torch.Tensor],
        label_name: str,
        label_dtype: torch.dtype,
        labels: Optional[List] = None,
        
    ) -> Dict[str, torch.Tensor]:
        """
        Add batch labels to the padded batch.

        Args:
            batch_padded (Dict[str, torch.Tensor]): The padded batch.
            batch_ids (Optional[List[str]]): A list of ids for the batch. 

        Returns:
            Dict[str, torch.Tensor]: The padded batch with labels added.
        """
        if labels is None:
            return data_dict
        
        if len(labels) != data_dict["taxa_ids"].shape[0]:
            raise ValueError(f"Batch labels length does not match the number of samples, {len(labels)} vs {data_dict['taxa_ids'].shape[0]}")
        
        data_dict[label_name] = torch.tensor(labels, dtype=label_dtype)
        return data_dict


    def tokenize_and_pad_batch(
        self,
        adata: ad.AnnData,
        batch_obskey: Optional[str] = None,
        continuous_obskey: Optional[str] = None,
        prepend_cls: bool = True,
        include_zero_count: bool = False,
        return_pt: bool = True,
        # mod_type: np.ndarray = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize and pad a batch of data. Returns a dict with padded taxa ids and values.

        Args:
            data (:class:`AnnData`):
            The binned data. size (num_samples, num_taxa)
            max_len (Optional[int]): The maximum length to pad/truncate to. If None, uses the max length in the batch.
        """
        tokenized_data = self.tokenize_batch(
            data=adata,
            return_pt=return_pt,
            prepend_cls=prepend_cls,
            include_zero_count=include_zero_count,
        )

        sample_dict = self.pad_batch(
            tokenized_data,
            cls_prepended=prepend_cls,
        )
        
        batch_ids = adata.obs[f"{batch_obskey}_id"].to_list() if batch_obskey else None
        sample_dict = self.add_labels(
            sample_dict,
            label_name="batch_labels",
            labels=batch_ids,
            label_dtype=torch.long,
        )
        
        if continuous_obskey:
            continuous_labels = adata.obs[continuous_obskey].to_list()
            sample_dict = self.add_labels(
                sample_dict,
                label_name="continuous_labels",
                labels=continuous_labels,
                label_dtype=torch.float32,
            )
            
        return sample_dict
    
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