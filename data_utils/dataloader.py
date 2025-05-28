from typing import List, Dict, Any
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from data_utils.tokenizer import MicrobiomeVocab

class SeqDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor], vocab: MicrobiomeVocab, gen_percent: float = 0, class_token: bool = True):
        """
        Args:
            data (Dict[str, torch.Tensor]): The input data.
            data should have these keys:
                - "taxa_ids": Tensor of shape (num_samples, seq_len)
                - "values": Tensor of shape (num_samples, seq_len)
        """
        self.data = data
        self.vocab = vocab
        self.gen_percent = gen_percent
        self.class_token = class_token

        if class_token:
            sample_length = self.data["taxa_ids"].shape[1] + 1
        else:
            sample_length = self.data["taxa_ids"].shape[1]

        if self.gen_percent > 0:
            self.gen_len = int(sample_length * self.gen_percent)
            self.pcpt_len = sample_length - self.gen_len
        else:
            raise NotImplementedError(
                "Generation percentage must be greater than 0 to use this dataset, for generation mode"
            )

    def __len__(self):
        return self.data["taxa_ids"].shape[0]

    def __getitem__(self, idx):
        # return {k: v[idx] for k, v in self.data.items()}
        return self.separate_pcpt_gen(
            ids=self.data["taxa_ids"][idx],
            values=self.data["values"][idx]
        )

    def separate_pcpt_gen(self, ids: torch.Tensor, values: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Step 1: Identify valid (non-class, non-pad) tokens
        valid_mask = (ids != self.vocab.pad_index) & (ids != self.vocab.class_index)
        valid_indices = torch.nonzero(valid_mask, as_tuple=True)[0]

        num_gen = int(len(valid_indices) * self.gen_percent)

        # Step 2: Randomly split into gen and pcpt
        perm = torch.randperm(len(valid_indices))
        gen_indices = valid_indices[perm[:num_gen]]
        pcpt_indices = valid_indices[perm[num_gen:]]

        gen_ids = ids[gen_indices]
        pcpt_ids = ids[pcpt_indices]

        gen_values = values[gen_indices]
        pcpt_values = values[pcpt_indices]

        # prepend class token to pcpt
        if self.class_token:
            pcpt_ids = torch.cat([torch.tensor([self.vocab.class_index], dtype=pcpt_ids.dtype), pcpt_ids])
            pcpt_values = torch.cat([torch.tensor([self.vocab.pad_value], dtype=pcpt_values.dtype), pcpt_values])

        # Step 3: Pad pcpt so that len(pcpt) == self.pcpt_len and len(gen) == self.gen_len
        gen_ids = torch.cat([gen_ids, torch.full((self.gen_len - len(gen_ids),), self.vocab.pad_index, dtype=gen_ids.dtype)])
        gen_values = torch.cat([gen_values, torch.full((self.gen_len - len(gen_values),), self.vocab.pad_value, dtype=gen_values.dtype)])
        pcpt_ids = torch.cat([pcpt_ids, torch.full((self.pcpt_len - len(pcpt_ids),), self.vocab.pad_index, dtype=pcpt_ids.dtype)])
        pcpt_values = torch.cat([pcpt_values, torch.full((self.pcpt_len - len(pcpt_values),), self.vocab.pad_value, dtype=pcpt_values.dtype)])

        return {
            "pcpt_ids": pcpt_ids,
            "pcpt_values": pcpt_values,
            "gen_ids": gen_ids,
            "gen_values": gen_values
        }


def prepare_dataloader(
    data_pt: Dict[str, torch.Tensor],
    batch_size: int,
    vocab: MicrobiomeVocab,
    shuffle: bool = False,
    gen_percent: float = 0.15,
    # intra_domain_shuffle: bool = False,
    drop_last: bool = False,
    num_workers: int = 0,
    # per_seq_batch_sample: bool = False,
) -> DataLoader:
    dataset = SeqDataset(data_pt, vocab, gen_percent=gen_percent)

    # # if per_seq_batch_sample, each batch will contain samples from the same experiment. Comment out for now because idk if we need this
    # if per_seq_batch_sample:
    #     # find the indices of samples in each seq batch
    #     subsets = []
    #     batch_labels_array = data_pt["batch_labels"].numpy()
    #     for batch_label in np.unique(batch_labels_array):
    #         batch_indices = np.where(batch_labels_array == batch_label)[0].tolist()
    #         subsets.append(batch_indices)
    #     data_loader = DataLoader(
    #         dataset=dataset,
    #         batch_sampler=SubsetsBatchSampler(
    #             subsets,
    #             batch_size,
    #             intra_subset_shuffle=intra_domain_shuffle,
    #             inter_subset_shuffle=shuffle,
    #             drop_last=drop_last,
    #         ),
    #         num_workers=num_workers,
    #         pin_memory=True,
    #     )
    #     return data_loader

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
    )
    return data_loader