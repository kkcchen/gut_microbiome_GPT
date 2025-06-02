from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch


@dataclass # syntactic sugar for defining a class with default values
class DataCollator:

    vocab: bool
    do_binning: bool = True
    do_padding: bool = True

    def __call__(
            self, examples: List[Dict[str, torch.Tensor]]
        ) -> Dict[str, torch.Tensor]:
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

            if len(self.reserve_keys) > 0:
                assert all(key in examples[0] for key in self.reserve_keys), (
                    f"reserve_keys must be a subset of the keys in the examples. "
                    f"Got {self.reserve_keys} but expected keys in {list(examples[0].keys())}."
                )

            # if self.data_style == "pcpt":
            #     data_dict = self._call_pcpt(examples)
            # elif self.data_style == "gen":
            #     data_dict = self._call_gen(examples)
            # elif self.data_style == "both":
            data_dict = self._call_both(examples)

            # add reserved keys
            device = examples[0]["genes"].device
            for key in self.reserve_keys:
                data_ = [example[key] for example in examples]
                data_dict[key] = torch.stack(data_, dim=0).to(device)

            return data_dict
    
    def _call_both(
        self,
        examples,
        probability: Optional[float] = None,
    ):
        """
        Args:
            examples (:obj:`List[Dict[str, torch.Tensor]]`): a list of data dicts.
                Each dict is for one cell. It contains multiple 1 dimensional tensors
                like the following exmaple:
                    {'taxa': tensor([36572, 17868, ..., 17072]),
                    'values': tensor([ 0.,  2., ..., 18.])}
            probability (float, optional): Probability of applying the transformation.
                Defaults to None.

        Returns:
            :obj:`Dict[str, torch.Tensor]`: a dict of tensors.
        """
        # Implement the logic for both PCPT and GEN data styles
        pass
        max_ori_len = max(len(example["taxa"]) for example in examples)
        _max_len = self.max_len if max_ori_len < self.max_len else max_ori_len
