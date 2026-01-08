from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset, DataLoader

from data_utils.vocab import MicrobiomeVocab
from data_utils.data_collator import DataCollator

class SeqDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        """
        Args:
            data (Dict[str, torch.Tensor]): The input data.
            data should have these keys:
                - "taxa_ids": Tensor of shape (num_samples, seq_len)
                - "values": Tensor of shape (num_samples, seq_len)
        """
        self.data = data
    
    def __len__(self):
        return self.data["taxa_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


def prepare_dataloader(
    data_pt: Dict[str, torch.Tensor],
    batch_size: int,
    vocab: MicrobiomeVocab,
    use_batch_labels: bool,
    use_continuous_labels: bool,
    mask_ids: bool = False,
    shuffle: bool = False,
    gen_percent: float = 0.15,
    # intra_domain_shuffle: bool = False,
    drop_last: bool = False,
    num_workers: int = 0,
    contrastive_embedding: bool = False,
    do_subsample: bool = False,
    do_clr: bool = False,
    # per_seq_batch_sample: bool = False,
) -> DataLoader:
    dataset = SeqDataset(data_pt)
    collator = DataCollator(
        vocab=vocab,
        sample_length=data_pt["taxa_ids"].shape[1],
        use_batch_labels=use_batch_labels,
        use_continuous_labels=use_continuous_labels,
        do_binning=True,
        do_padding=True,
        mask_ids=mask_ids,
        gen_percent=gen_percent,
        use_class_token=True,
        contrastive_embedding=contrastive_embedding,
        do_subsample=do_subsample,
        do_clr=do_clr,
    )

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
        collate_fn=collator,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
    )
    return data_loader