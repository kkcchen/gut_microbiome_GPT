"""
Vocabulary classes for taxa and batch labels.
"""
from pathlib import Path
import pickle
import numpy as np
import anndata as ad
from typing import List, Union


class TaxaVocabulary:
    """
    Vocabulary for taxonomic features.
    Maps taxa names to integer indices.
    """
    
    def __init__(self, taxa_names: List[str]):
        """
        Initialize vocabulary from taxa names.
        
        :param taxa_names: List of taxa names from training data.
        """
        self.taxa_names = taxa_names
        self.token_to_id = {name: idx for idx, name in enumerate(taxa_names)}
        self.id_to_token = {idx: name for idx, name in enumerate(taxa_names)}
    
    @classmethod
    def from_adata(cls, adata: ad.AnnData) -> 'TaxaVocabulary':
        """
        Build vocabulary from AnnData object.
        
        :param adata: AnnData with taxa names in .var_names.
        :return: TaxaVocabulary instance.
        """
        taxa_names = adata.var_names.tolist()
        return cls(taxa_names)
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.taxa_names)
    
    def encode_taxa(self, taxa_names: List[str]) -> np.ndarray:
        """
        Encode list of taxa names to integer IDs.
        
        :param taxa_names: List of taxa names.
        :return: Array of integer IDs.
        """
        return np.array([self.token_to_id[name] for name in taxa_names])
    
    def decode_taxa(self, taxa_ids: np.ndarray) -> List[str]:
        """
        Decode integer IDs to taxa names.
        
        :param taxa_ids: Array of integer IDs.
        :return: List of taxa names.
        """
        return [self.id_to_token[int(idx)] for idx in taxa_ids]
    
    def add_taxa(self, new_taxa_name: str, new_idx: int):
        """
        Add new taxa to the vocabulary.
        
        :param new_taxa_name: List of new taxa names to add.
        """
        assert new_taxa_name not in self.token_to_id, f"Taxa '{new_taxa_name}' already exists in the vocabulary"
        assert new_idx not in self.id_to_token, f"Index '{new_idx}' already exists in the vocabulary"
        self.taxa_names.append(new_taxa_name)
        self.token_to_id[new_taxa_name] = new_idx
        self.id_to_token[new_idx] = new_taxa_name
    
    def get_taxa_name(self, idx: int) -> str:
        """Get taxa name by index."""
        return self.id_to_token[idx]
    
    def get_taxa_id(self, name: str) -> int:
        """Get taxa index by name."""
        return self.token_to_id[name]
    
    def save(self, path: Union[str, Path]):
        """
        Save vocabulary to disk.
        
        :param path: Path to save vocabulary (e.g., 'taxa_vocab.pkl')
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
            
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'TaxaVocabulary':
        """
        Load vocabulary from disk.
        
        :param path: Path to vocabulary file.
        :return: Loaded TaxaVocabulary instance.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {path}")
        with open(path, 'rb') as f:
            vocab = pickle.load(f)
        return vocab


class BatchVocabulary:
    """
    Vocabulary for batch/study labels.
    """
    
    def __init__(self, batch_names: List[str]):
        """
        Initialize batch vocabulary.
        
        :param batch_names: List of unique batch/study names.
        """
        self.batch_names = sorted(set(batch_names)) + ['unknown']
        self.token_to_id = {name: idx for idx, name in enumerate(self.batch_names)}
        self.id_to_token = {idx: name for idx, name in enumerate(self.batch_names)}
        self.unk_id = len(self.batch_names) - 1  # ID for 'unknown' batch
    
    @classmethod
    def from_adata(cls, adata: ad.AnnData, key: str = 'study_id') -> 'BatchVocabulary':
        """
        Build batch vocabulary from AnnData metadata.
        
        :param adata: AnnData with batch labels in .obs[key].
        :param key: Column name in adata.obs for batch labels.
        :return: BatchVocabulary instance.
        """
        batch_names = adata.obs[key].unique().tolist()
        return cls(batch_names)
    
    def __len__(self) -> int:
        """Return number of unique batches."""
        return len(self.batch_names)
    
    def encode_batches(self, batch_names: List[str]) -> np.ndarray:
        """
        Encode list of batch names to integer IDs.
        
        :param batch_names: List of batch names.
        :return: Array of integer IDs.
        """
        encoded = []
        
        for name in batch_names:
            if name in self.token_to_id:
                encoded.append(self.token_to_id[name])
            else:
                # Map unknown batch to <UNK> token
                encoded.append(self.unk_id)        
        return np.array(encoded)
    
    def decode_batches(self, batch_ids: np.ndarray) -> List[str]:
        """
        Decode integer IDs to batch names.
        
        :param batch_ids: Array of integer IDs.
        :return: List of batch names.
        """
        return [self.id_to_token[int(idx)] for idx in batch_ids]

    def save(self, path: Union[str, Path]):
        """
        Save vocabulary to disk.
        
        :param path: Path to save vocabulary (e.g., 'batch_vocab.pkl')
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
            
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'BatchVocabulary':
        """
        Load vocabulary from disk.
        
        :param path: Path to vocabulary file.
        :return: Loaded BatchVocabulary instance.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {path}")
        with open(path, 'rb') as f:
            vocab = pickle.load(f)
        return vocab

