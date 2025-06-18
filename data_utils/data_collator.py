from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from data_utils.vocab import MicrobiomeVocab

import torch


@dataclass # syntactic sugar for defining a class with default values
class DataCollator:
    def __init__(self, vocab: MicrobiomeVocab, sample_length: int, use_batch_labels: bool, do_binning: bool = True, do_padding: bool = True, gen_percent: float = 0.15, use_class_token: bool = True, contrastive_embedding: bool = False):
        """
        Initializes the data collator with specified parameters.

        Args:
            vocab (MicrobiomeVocab): The vocabulary object used for encoding microbiome data.
            sample_length (int): The total length of the sample, which must include space for the class token if `use_class_token` is set to True.
            do_binning (bool, optional): Whether to perform binning on the data. Defaults to True.
            do_padding (bool, optional): Whether to pad the data to the specified sample length. Defaults to True.
            gen_percent (float, optional): The percentage of the sample length to be used for generation mode. Defaults to 0.2.
            use_class_token (bool, optional): Whether to include a class token in the sample. Defaults to True.

        Attributes:
            vocab (MicrobiomeVocab): The vocabulary object used for encoding microbiome data.
            do_binning (bool): Indicates whether binning is enabled.
            do_padding (bool): Indicates whether padding is enabled.
            gen_percent (float): The percentage of the sample length used for generation mode.
            use_class_token (bool): Indicates whether a class token is included in the sample.
            generation_mode (bool): Indicates whether generation mode is enabled based on `gen_percent`.
            gen_len (int): The length of the generated portion of the sample, calculated as `sample_length * gen_percent`.
            pcpt_len (int): The length of the perceptive portion of the sample, calculated as `sample_length - gen_len`.
        """
        self.vocab = vocab
        self.do_binning = do_binning
        self.do_padding = do_padding
        self.gen_percent = gen_percent
        self.use_class_token = use_class_token
        self.use_batch_labels = use_batch_labels
        self.contrastive_embedding = contrastive_embedding

        if self.gen_percent > 0:
            self.generation_mode = True
            self.gen_len = int(sample_length * self.gen_percent)
            self.pcpt_len = sample_length - self.gen_len
        else:
            self.generation_mode = False
        
        if not self.generation_mode:
            assert not self.contrastive_embedding, "Contrastive embedding is only supported in generation mode."

    def __call__(
            self, examples: List[Dict[str, torch.Tensor]]
        ) -> Dict:
            """
            Args:
                examples (:obj:`List[Dict[str, torch.Tensor]]`): a list of data dicts.
                    Each dict is for one cell. It contains multiple 1 dimensional tensors
                    like the following exmaple:
                        {'id': tensor(184117),
                        'taxa': tensor([36572, 17868, ..., 17072]),
                        'values': tensor([ 0.,  2., ..., 18.])}

            Returns:
                :obj:`Dict[str, torch.Tensor]`: a dict of tensors.
            """

            ids_batch = torch.stack([example["taxa_ids"] for example in examples])
            values_batch = torch.stack([example["values"] for example in examples])
            batch_labels = torch.tensor([example["batch_labels"] for example in examples], dtype=torch.long) if self.use_batch_labels else None
            
            if self.generation_mode:
                if self.contrastive_embedding:
                    view1 = self.separate_pcpt_gen(ids_batch, values_batch)
                    view2 = self.separate_pcpt_gen(ids_batch, values_batch)
                    out_dict = {
                        "view1": view1,
                        "view2": view2,
                    }
                else:
                    out_dict = self.separate_pcpt_gen(ids_batch, values_batch)
            else:
                out_dict = {
                    "ids": ids_batch,
                    "values": values_batch
                }

            if self.use_batch_labels:
                out_dict["batch_labels"] = batch_labels

            return out_dict

    # def _call_both(
    #     self,
    #     examples,
    #     probability: Optional[float] = None,
    # ):
    #     """
    #     Args:
    #         examples (:obj:`List[Dict[str, torch.Tensor]]`): a list of data dicts.
    #             Each dict is for one cell. It contains multiple 1 dimensional tensors
    #             like the following exmaple:
    #                 {'taxa': tensor([36572, 17868, ..., 17072]),
    #                 'values': tensor([ 0.,  2., ..., 18.])}
    #         probability (float, optional): Probability of applying the transformation.
    #             Defaults to None.

    #     Returns:
    #         :obj:`Dict[str, torch.Tensor]`: a dict of tensors.
    #     """
    #     # Implement the logic for both PCPT and GEN data styles
    #     pass
    #     max_ori_len = max(len(example["taxa"]) for example in examples)
    #     _max_len = self.max_len if max_ori_len < self.max_len else max_ori_len
        

    def separate_pcpt_gen(self, ids: torch.Tensor, values: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, T = ids.shape
        device = ids.device

        # Step 1: Create a mask for valid positions (not pad or class)
        valid_mask = (ids != self.vocab.pad_index) & (ids != self.vocab.class_index)

        # Prepare tensors to hold the results
        gen_ids = torch.full((B, self.gen_len), self.vocab.pad_index, dtype=ids.dtype, device=device)
        gen_values = torch.full((B, self.gen_len), self.vocab.pad_value, dtype=values.dtype, device=device)
        pcpt_ids = torch.full((B, self.pcpt_len), self.vocab.pad_index, dtype=ids.dtype, device=device)
        pcpt_values = torch.full((B, self.pcpt_len), self.vocab.pad_value, dtype=values.dtype, device=device)

        for i in range(B):
            valid_indices = torch.nonzero(valid_mask[i], as_tuple=True)[0]
            total_valid = len(valid_indices)

            if total_valid == 0:
                continue  # skip empty rows

            gen_len = min(int(total_valid * self.gen_percent), self.gen_len)
            pcpt_len = self.pcpt_len - (1 if self.use_class_token else 0)

            perm = torch.randperm(total_valid, device=device)
            gen_idx = valid_indices[perm[:gen_len]]
            pcpt_idx = valid_indices[perm[gen_len:gen_len + pcpt_len]]

            # Get actual values
            gen_ids[i, :len(gen_idx)] = ids[i, gen_idx]
            gen_values[i, :len(gen_idx)] = values[i, gen_idx]

            pcpt_ids_i = ids[i, pcpt_idx]
            pcpt_values_i = values[i, pcpt_idx]

            if self.use_class_token:
                pcpt_ids[i, 0] = self.vocab.class_index
                pcpt_values[i, 0] = self.vocab.pad_value
                pcpt_ids[i, 1:1 + len(pcpt_idx)] = pcpt_ids_i
                pcpt_values[i, 1:1 + len(pcpt_idx)] = pcpt_values_i
            else:
                pcpt_ids[i, :len(pcpt_idx)] = pcpt_ids_i
                pcpt_values[i, :len(pcpt_idx)] = pcpt_values_i

        return {
            "pcpt_ids": pcpt_ids,
            "pcpt_values": pcpt_values,
            "gen_ids": gen_ids,
            "gen_values": gen_values
        }