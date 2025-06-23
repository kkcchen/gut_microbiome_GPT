import numpy as np
from typing import Dict, Optional, Union, List, Tuple
import json

class MicrobiomeVocab():
    def __init__(
        self,
        vocab_list: List[str],
        class_token: str = "<cls>",
        mask_token: str = "<mask>",
        pad_token: str = "<pad>",
        pad_value: int = 0,
        mask_value: int = -1,
        add_special_tokens: bool = True
    ):
        """
        Initialize the vocabulary with taxa and special tokens.
        Args:
            vocab_list (List[str]): A list of taxa names.
            class_token (str): The token representing the class.
            mask_token (str): The token representing the mask.
        """
        if add_special_tokens:
            self.itos = vocab_list + [class_token, mask_token, pad_token]
        else:
            self.itos = vocab_list
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
    
    def save_vocab_json(self, path, meta_path):
        """
        Save the vocabulary to a JSON file.
        Args:
            path (str): The path to save the JSON file.
            meta_path (str): The path to save the metadata file.
        """
        with open(path, 'w') as f:
            json.dump(self.stoi, f, indent=4)
            
        metadata = {
        "class_token": self.class_token,
        "mask_token": self.mask_token,
        "pad_token": self.pad_token,
        "pad_value": self.pad_value,
        "mask_value": self.mask_value,
        }

        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=4)
    
    @classmethod 
    def get_vocab_from_json(cls, path, meta_path) -> 'MicrobiomeVocab':
        """
        Load the vocabulary from a JSON file.
        Args:
            path (str): The path to the JSON file.
        Returns:
            MicrobiomeVocab: An instance of MicrobiomeVocab with the loaded vocabulary.
        """

        # Load vocabulary
        with open(path, 'r') as f:
            vocab_dict = json.load(f)

        # Load metadata
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
        
        sorted_items = sorted(vocab_dict.items(), key=lambda item: item[1])
        
        vocab_list = [item[0] for item in sorted_items]
        return cls(
            vocab_list=vocab_list,
            class_token=metadata["class_token"],
            mask_token=metadata["mask_token"],
            pad_token=metadata["pad_token"],
            pad_value=metadata["pad_value"],
            mask_value=metadata["mask_value"],
            add_special_tokens=False
        )

class BatchVocab():
    """
    A class to represent the vocabulary of batches in the dataset.
    """

    def __init__(self, vocab, keep_order: bool = False):
        """
        Initialize the vocabulary with taxa and special tokens.

        Args:
            vocab (np.ndarray): A numpy array containing batch names. 
                The first column should be the sample names, and the rest are batch names
        """
        # get the unique batch names
        if not keep_order:
            vocab_set = set(vocab)
            self.itos = list(vocab_set)
        else:
            self.itos = vocab
        self.stoi = {token: idx for idx, token in enumerate(self.itos)}
        
    def __getitem__(self, item: str):
        return self.stoi.get(item, -1)
    
    def save_vocab_json(self, path):
        """
        Save the vocabulary to a JSON file.
        Args:
            path (str): The path to save the JSON file.
        """
        with open(path, 'w') as f:
            json.dump(self.stoi, f, indent=4)
            
    def __len__(self):
        return len(self.itos)
    
    @classmethod
    def get_vocab_from_json(cls, path) -> 'BatchVocab':
        """
        Load the vocabulary from a JSON file.
        Args:
            path (str): The path to the JSON file.
        Returns:
            BatchVocab: An instance of BatchVocab with the loaded vocabulary.
        """
        with open(path, 'r') as f:
            vocab_dict = json.load(f)
        
        vocab_list = [item[0] for item in sorted(vocab_dict.items(), key=lambda item: item[1])]
        assert len(vocab_list) == len(vocab_dict), "Duplicate batch names found in the JSON file."
        return cls(vocab_list, keep_order=True)
