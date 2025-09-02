import numpy as np
from typing import Dict, Optional, Union, List, Tuple
import json
import anndata as ad

class MicrobiomeVocab():
    def __init__(
        self,
        vocab_list,
        class_token: str = "<cls>",
        mask_token: str = "<mask>",
        pad_token: str = "<pad>",
        pad_value: int = 0,
        mask_value: int = -1,
    ):
        """
        Initialize the vocabulary with taxa and special tokens.
        Args:
            vocab_list (List[str]): A list of taxa names.
            class_token (str): The token representing the class.
            mask_token (str): The token representing the mask.
        """
        # IMPORTANT: special tokens must be at end of the vocabulary list
        special_tokens = [class_token, mask_token, pad_token]
        self.itos = vocab_list + special_tokens
        self.num_special_tokens = len(special_tokens)

        # assert there are no duplicates in the taxa names
        if len(self.itos) != len(set(self.itos)):
            raise ValueError("Duplicate taxa names found in the DataFrame.")
        
        self.stoi = {token: idx for idx, token in enumerate(self.itos)}
        self.class_token = class_token
        self.mask_token = mask_token
        self.pad_token = pad_token

        self.pad_value = pad_value
        self.mask_value = mask_value

        self.pad_index = self.stoi[pad_token]
        self.mask_index = self.stoi[mask_token]
        self.class_index = self.stoi[class_token]

    def __len__(self):
        return len(self.itos)

    def __getitem__(self, item: str):
        return self.stoi.get(item, self.pad_index)
    
    def lookup_indices(self, taxa_names: List[str]) -> List[int]:
        """
        Lookup the indices of the taxa_ids in the vocabulary.
        Args:
            taxa_names (List[str]): A list of taxa names.
        Returns:
            List[int]: A list of indices of the taxa_ids in the vocabulary.
        """
        return [self[taxa_name] for taxa_name in taxa_names]
    
    @classmethod 
    def create_vocab_from_scratch(
        cls,
        adata: ad.AnnData,
        class_token: str = "<cls>",
        mask_token: str = "<mask>",
        pad_token: str = "<pad>",
        pad_value: int = 0,
        mask_value: int = -1,
        ) -> 'MicrobiomeVocab':
        assert "taxa" in adata.var, "The AnnData object must have 'taxa' in var."
        vocab_list = adata.var["taxa"].tolist()
        
        vocab = cls(
            vocab_list=vocab_list,
            class_token=class_token,
            mask_token=mask_token,
            pad_token=pad_token,
            pad_value=pad_value,
            mask_value=mask_value,
        )
        
        # add indices as a adata.var column
        adata.var["taxa_id"] = adata.var_names.map(vocab.stoi)
        adata.uns["vocab_metadata"] = {
            "class_token": class_token,
            "mask_token": mask_token,
            "pad_token": pad_token,
            "pad_value": pad_value,
            "mask_value": mask_value,
        }
        
        return vocab
    
    
    @classmethod 
    def restore_vocab(cls, adata) -> 'MicrobiomeVocab':
        """
        Load the vocabulary.
        Args:
            adata
        Returns:
            MicrobiomeVocab: An instance of MicrobiomeVocab with the loaded vocabulary.
        """
        
        assert "vocab_metadata" in adata.uns, "The AnnData object must have 'vocab_metadata' in uns."
        assert "taxa_id" in adata.var, "The AnnData object must have 'taxa_id' in var."
        metadata = adata.uns["vocab_metadata"]

        return cls(
            vocab_list=adata.var_names.tolist(),
            class_token=metadata["class_token"],
            mask_token=metadata["mask_token"],
            pad_token=metadata["pad_token"],
            pad_value=metadata["pad_value"],
            mask_value=metadata["mask_value"],
        )

class BatchVocab():
    """
    A class to represent the vocabulary of batches in the dataset.
    """

    def __init__(self, vocab):
        """
        Initialize the vocabulary with taxa and special tokens.

        Args:
            vocab (np.ndarray): A numpy array containing batch names. 
                The first column should be the sample names, and the rest are batch names
        """
        assert len(vocab) == len(set(vocab)), "Duplicate batch names found in the vocabulary."
        self.itos = vocab.tolist()
        self.stoi = {token: idx for idx, token in enumerate(self.itos)}
        
    def __getitem__(self, item: str):
        return self.stoi.get(item, -1)
            
    def __len__(self):
        return len(self.itos)
    
    @classmethod
    def create_batchvocab_from_scratch(cls, batch_obskey, adata) -> 'BatchVocab':
        assert batch_obskey in adata.obs, f"The AnnData object must have '{batch_obskey}' in obs."
        batch_vocab = cls(
            vocab=adata.obs[batch_obskey].unique(),
        )
        
        adata.obs[f"{batch_obskey}_id"] = adata.obs[batch_obskey].map(batch_vocab.stoi)
        adata.uns[f"{batch_obskey}_batch_vocab"] = batch_vocab.itos
        return batch_vocab

    
    @classmethod
    def restore_batchvocab(cls, adata, batch_obskey) -> 'BatchVocab':
        """
        Load the vocabulary from an anndata
        Args:
            adata (ad.AnnData): The AnnData object containing the batch vocabulary.
        """
        assert f"{batch_obskey}_batch_vocab" in adata.uns, "The AnnData object must have 'batch_vocab' in uns."
        vocab_list = adata.uns[f"{batch_obskey}_batch_vocab"]
        return cls(vocab_list)
