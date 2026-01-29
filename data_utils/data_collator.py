from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from data_utils.vocab import MicrobiomeVocab
import time

import torch

class DataCollator:
    def __init__(self, vocab: MicrobiomeVocab, sample_length: int, use_batch_labels: bool, use_continuous_labels: bool, 
                 mask_ids: bool = False, do_binning: bool = True, do_padding: bool = True, gen_percent: float = 0.15, 
                 use_class_token: bool = True, contrastive_embedding: bool = False, do_subsample: bool = False, do_clr: bool = False):
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
        self.use_continuous_labels = use_continuous_labels
        self.contrastive_embedding = contrastive_embedding
        self.mask_ids = mask_ids
        
        self.do_subsample = do_subsample
        self.do_clr = do_clr #####

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
                        {'taxa_ids': tensor(184117),
                        'values': tensor([36572, 17868, ..., 17072]),
                        'batch_labels': tensor([ 0.,  2., ..., 18.])}

            Returns:
                :obj:`Dict[str, torch.Tensor]`: a dict of tensors.
            """

            ids_batch = torch.stack([example["taxa_ids"] for example in examples])
            values_batch = torch.stack([example["values"] for example in examples])

            # attempt at subsampling noising
            if self.do_subsample:
                
                # Safety checks
                if not torch.all(values_batch >= 0):
                    raise ValueError("Negative values found in values_batch; cannot cast to counts.")

                # Optional: check near-integer floats
                if not torch.allclose(values_batch, values_batch.round(), atol=1e-6):
                    raise ValueError("values_batch contains non-integer floats; cannot safely cast to int.")

                values_batch = values_batch.round().to(torch.int64)
                timer = time.time()
                view1, view2 = self.make_downsampled_views(ids_batch, values_batch)
                print("Subsampling time:", time.time() - timer)
                out_dict = {"view1": view1, "view2": view2}
###############################                
                if "obs_idx" in examples[0]:
                    out_dict["obs_idx"] = torch.stack([example["obs_idx"] for example in examples])  # shape (B,)

            
            elif self.generation_mode:
                if self.contrastive_embedding:
                    view1 = self.mlm_corrupt_values(ids_batch, values_batch, self.mask_ids)
                    view2 = self.mlm_corrupt_values(ids_batch, values_batch, self.mask_ids)
                    out_dict = {
                        "view1": view1,
                        "view2": view2,
                    }
                else:
                    out_dict = self.mlm_corrupt_values(ids_batch, values_batch, self.mask_ids)
            else:
                out_dict = {
                    "ids": ids_batch,
                    "values": values_batch
                }
###############################                
                if "obs_idx" in examples[0]:
                    out_dict["obs_idx"] = torch.stack([example["obs_idx"] for example in examples])  # shape (B,)

                
            if self.do_clr:
                out_dict['values'] = self.clr_torch(out_dict['ids'], out_dict['values'])
                # B = ids_batch.shape[0]
                # for i in range(B):
                #     out_dict['values'][i] = self.clr_torch(out_dict['ids'][i], out_dict['values'][i])
###############################                
                if "obs_idx" in examples[0]:
                    out_dict["obs_idx"] = torch.stack([example["obs_idx"] for example in examples])  # shape (B,)


            if self.use_batch_labels:
                out_dict["batch_labels"] = torch.tensor([example["batch_labels"] for example in examples], dtype=torch.long)
            
            if self.use_continuous_labels:
                out_dict["continuous_labels"] = torch.tensor([example["continuous_labels"] for example in examples], dtype=torch.float)

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
        

    # def separate_pcpt_gen(self, ids: torch.Tensor, values: torch.Tensor) -> Dict[str, torch.Tensor]:
    #     B, T = ids.shape
    #     device = ids.device

    #     # Step 1: Create a mask for valid positions (not pad or class)
    #     valid_mask = (ids != self.vocab.pad_index) & (ids != self.vocab.class_index)

    #     # Prepare tensors to hold the results
    #     gen_ids = torch.full((B, self.gen_len), self.vocab.pad_index, dtype=ids.dtype, device=device)
    #     gen_values = torch.full((B, self.gen_len), self.vocab.pad_value, dtype=values.dtype, device=device)
    #     pcpt_ids = torch.full((B, self.pcpt_len), self.vocab.pad_index, dtype=ids.dtype, device=device)
    #     pcpt_values = torch.full((B, self.pcpt_len), self.vocab.pad_value, dtype=values.dtype, device=device)

    #     for i in range(B):
    #         valid_indices = torch.nonzero(valid_mask[i], as_tuple=True)[0]
    #         total_valid = len(valid_indices)

    #         if total_valid == 0:
    #             continue  # skip empty rows

    #         gen_len = min(int(total_valid * self.gen_percent), self.gen_len)
    #         pcpt_len = self.pcpt_len - (1 if self.use_class_token else 0)

    #         perm = torch.randperm(total_valid, device=device)
    #         gen_idx = valid_indices[perm[:gen_len]]
    #         pcpt_idx = valid_indices[perm[gen_len:gen_len + pcpt_len]]

    #         # Get actual values
    #         gen_ids[i, :len(gen_idx)] = ids[i, gen_idx]
    #         gen_values[i, :len(gen_idx)] = values[i, gen_idx]

    #         pcpt_ids_i = ids[i, pcpt_idx]
    #         pcpt_values_i = values[i, pcpt_idx]

    #         if self.use_class_token:
    #             pcpt_ids[i, 0] = self.vocab.class_index
    #             pcpt_values[i, 0] = self.vocab.pad_value
    #             pcpt_ids[i, 1:1 + len(pcpt_idx)] = pcpt_ids_i
    #             pcpt_values[i, 1:1 + len(pcpt_idx)] = pcpt_values_i
    #         else:
    #             pcpt_ids[i, :len(pcpt_idx)] = pcpt_ids_i
    #             pcpt_values[i, :len(pcpt_idx)] = pcpt_values_i

    #     return {
    #         "pcpt_ids": pcpt_ids,
    #         "pcpt_values": pcpt_values,
    #         "gen_ids": gen_ids,
    #         "gen_values": gen_values
    #     }
    def _log_depth(self, ids: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """
        ids: (B, L) or (L,)
        counts: (B, L) or (L,) raw integer counts (>=0)

        Returns: (B,) or () log1p(depth) computed over valid (non-pad, non-class) taxa.
        """
        valid = (ids != self.vocab.pad_index) & (ids != self.vocab.class_index)
        depth = (counts.to(torch.float32) * valid.to(torch.float32)).sum(dim=-1)  # sums over L
        return torch.log1p(depth)    

    def mlm_corrupt_values(self, ids: torch.Tensor, values: torch.Tensor, mask_ids=False) -> Dict:
        B, T = ids.shape
        device = ids.device

        # Clone inputs to avoid in-place modification
        corrupted_values = values.clone()
        corrupted_ids = ids.clone()
        target_values = torch.where(
            ids != self.vocab.pad_index,
            torch.tensor(float(self.vocab.mask_value), device=ids.device),
            torch.tensor(float(self.vocab.pad_value), device=ids.device),
        ) # later, known_positions = values_target.eq(vocab.mask_value)
        target_ids = torch.full(
            ids.shape,
            self.vocab.pad_index,
            dtype=torch.long,
            device=ids.device,
        )

        # Mask to find valid (non-pad, non-class) positions
        valid_mask = (ids != self.vocab.pad_index) & (ids != self.vocab.class_index)

        for i in range(B):
            valid_indices = torch.nonzero(valid_mask[i], as_tuple=True)[0]
            total_valid = len(valid_indices)

            if total_valid == 0:
                continue

            num_to_mask = max(2, int(total_valid * self.gen_percent))
            perm = torch.randperm(total_valid, device=device)
            mask_indices = valid_indices[perm[:num_to_mask]]

            probs = torch.rand(len(mask_indices), device=device)

            if mask_ids:
                # probs: tensor of shape (N,)
                N = probs.shape[0]
                half = N // 2

                # Split into halves
                probs_val = probs[:half]
                probs_id = probs[half:]
                
                val_mask_indices = mask_indices[:half]
                id_mask_indices = mask_indices[half:]

                ### ---- Value-based masking ----
                mask_mask = probs_val < 0.8                        # 80% replace with [MASK] value
                rand_mask = (probs_val >= 0.8) & (probs_val < 0.9) # 10% replace with random value
                unchanged_mask = probs_val >= 0.9                  # 10% unchanged

                ### ---- ID-based masking ----
                id_mask_mask = probs_id < 0.8                      # 80% replace id with [MASK] id
                id_rand_mask = (probs_id >= 0.8) & (probs_id < 0.9)# 10% replace id with random id
                id_unchanged_mask = probs_id >= 0.9                # 10% unchanged
            
                
                # save original ids for target
                target_ids[i, id_mask_indices[id_mask_mask | id_rand_mask | id_unchanged_mask]] = ids[i, id_mask_indices[id_mask_mask | id_rand_mask | id_unchanged_mask]]
            else:
                val_mask_indices = mask_indices
                # 80% replace with [MASK] value
                mask_mask = probs < 0.8
                # 10% replace with random value
                rand_mask = (probs >= 0.8) & (probs < 0.9)
                # 10% unchanged (do nothing)
                unchanged_mask = (probs >= 0.9)
            
            # save original values for target
            target_values[i, val_mask_indices[mask_mask | rand_mask | unchanged_mask]] = values[i, val_mask_indices[mask_mask | rand_mask | unchanged_mask]]

            if mask_mask.any():
                corrupted_values[i, val_mask_indices[mask_mask]] = self.vocab.mask_value

            if rand_mask.any():
                random_values = torch.randint(
                    low=int(values[i].min().item()),
                    high=int(values[i].max().item()) + 1,
                    size=(rand_mask.sum().item(),),
                    dtype=values.dtype,
                    device=device
                )
                corrupted_values[i, val_mask_indices[rand_mask]] = random_values

            if mask_ids:
                if id_mask_mask.any():
                    corrupted_ids[i, id_mask_indices[id_mask_mask]] = self.vocab.mask_index

                if id_rand_mask.any():
                    all_indices = torch.arange(len(self.vocab), device=device)
                    valid_indices = all_indices[(all_indices != self.vocab.pad_index) & (all_indices != self.vocab.class_index)]

                    # Randomly sample from valid indices
                    random_ids = valid_indices[torch.randint(
                        low=0,
                        high=valid_indices.size(0),
                        size=(id_rand_mask.sum().item(),),
                        dtype=torch.long,
                        device=device
                    )]
                    corrupted_ids[i, id_mask_indices[id_rand_mask]] = random_ids

        return {
            "ids": corrupted_ids,
            "target_ids": target_ids,
            "corrupted_values": corrupted_values,
            "target_values": target_values,
        }


    @torch.no_grad()
    def subsample_counts_torch(
        self,
        ids: torch.Tensor,
        counts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Binomial downsampling of counts across batch

        - Only subsamples valid taxa positions (non-pad, non-class).
        - Keeps pad/class positions unchanged.
        - If total counts <= subsample_target, returns original counts.
        """
        device = counts.device


        # Raw-count strictness: integer dtype + nonnegative
        if counts.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise TypeError(
                f"Expected raw integer counts tensor, got dtype={counts.dtype}. "
                "Pass raw counts here (before any binning/normalization)."
            )
        if torch.any(counts < 0):
            raise ValueError("Counts must be nonnegative.")

        valid_mask = (ids != self.vocab.pad_index) & (ids != self.vocab.class_index)
        if not torch.any(valid_mask):
            print("Warning: No valid taxa positions found for subsampling.")
            return counts

        B, L = counts.shape

        # valid_counts = counts[valid_mask]

        p = torch.rand(B, device=device) + 1e-8
        p = torch.clamp(p, max=1.0)

        downsampled = torch.binomial(counts.float(), p[:, None]).to(counts.dtype)

        out = counts.clone()
        out[valid_mask] = downsampled[valid_mask]
        return out


    @torch.no_grad()
    def subsample_counts_old(
        self,
        ids_1d: torch.Tensor,
        counts_1d: torch.Tensor,
    ) -> torch.Tensor:
        """
        Multinomial downsampling of sample's raw counts.

        - Only subsamples valid taxa positions (non-pad, non-class).
        - Keeps pad/class positions unchanged.
        - If total counts <= subsample_target, returns original counts.
        """
        device = counts_1d.device


        # Raw-count strictness: integer dtype + nonnegative
        if counts_1d.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise TypeError(
                f"Expected raw integer counts tensor, got dtype={counts_1d.dtype}. "
                "Pass raw counts here (before any binning/normalization)."
            )
        if torch.any(counts_1d < 0):
            raise ValueError("Counts must be nonnegative.")

        counts = counts_1d.clone()

        valid_mask = (ids_1d != self.vocab.pad_index) & (ids_1d != self.vocab.class_index)
        if not torch.any(valid_mask):
            print("Warning: No valid taxa positions found for subsampling.")
            return counts

        valid_counts = counts[valid_mask]
        total = int(valid_counts.sum().item())
        
        if total < 1000:  # ==  0: # Already all zeros across valid positions
            
            return counts

        subsample_target = int(torch.randint(1000, total+1,(1,),device=device).item())

        probs = valid_counts.to(torch.float)
        probs = probs / probs.sum()

        sampled = torch.distributions.Multinomial(
            total_count=subsample_target, probs=probs
        ).sample().to(valid_counts.dtype)

        out = counts
        out[valid_mask] = sampled
        return out

    @torch.no_grad()
    def make_downsampled_views(
        self,
        ids: torch.Tensor,
        values: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Two downsampled views for contrastive learning.
        Each view is dict: {"ids": ids, "values": downsampled_counts}
        """
        # B, L = values.shape
        # # v1 = torch.empty_like(values)
        # # v2 = torch.empty_like(values)   
        # v1 = torch.zeros(values.shape, device=values.device, dtype=torch.float32)
        # v2 = torch.zeros(values.shape, device=values.device, dtype=torch.float32)

        # for i in range(B):
        #     timer = time.time()
        #     c1 = self.subsample_counts_torch(ids[i], values[i])
        #     c2 = self.subsample_counts_torch(ids[i], values[i])
        #     print("Subsample time per sample:", time.time() - timer)
        #     v1[i] = self.clr_torch(ids[i], c1)
        #     v2[i] = self.clr_torch(ids[i], c2)

        # view1 = {"ids": ids.clone(), "values": v1}
        # view2 = {"ids": ids.clone(), "values": v2}
        # return view1, view2
        
        
        c1 = self.subsample_counts_torch(ids, values)
        c2 = self.subsample_counts_torch(ids, values)
        
        # Auxiliary target computed from the SAME subsampled counts used for each view
        log_depth_1 = self._log_depth(ids, c1)  # (B,)
        log_depth_2 = self._log_depth(ids, c2)  # (B,)

        v1 = self.clr_torch(ids, c1)  # (B, L) float32
        v2 = self.clr_torch(ids, c2)

        return {"ids": ids.clone(), "values": v1}, {"ids": ids.clone(), "values": v2} #, "log_depth": log_depth_1}, \, "log_depth": log_depth_2}

    @torch.no_grad()    
    def clr_torch(self, ids: torch.Tensor, counts: torch.Tensor, pseudocount: float = 1e-6):
        # ids_1d: (T,), counts_1d: (T,) integer counts
        valid = (ids != self.vocab.pad_index) & (ids != self.vocab.class_index)

        x = counts.to(torch.float32)
        out = torch.zeros_like(x)
        
        
        # modify this to just return the logged relative abundance instead of clr
        # xv = torch.where(valid, x + pseudocount, torch.zeros_like(x))  # (B,L)

        # denom = xv.sum(dim=1, keepdim=True).clamp_min(pseudocount)     # (B,1)

        # out = torch.zeros_like(xv)
        # out_valid = torch.log(xv.clamp_min(pseudocount)) - torch.log(denom)  # (B,L) via broadcast
        # out[valid] = out_valid[valid]
        
        # Add pseudocount only where valid
#########
        # xv = torch.where(valid, x + pseudocount, torch.ones_like(x))  

        # logx = torch.log(xv)  # (B, L)

        # # Per-sample mean of log counts over valid positions (matches logx.mean() on xv[valid] in 1D)
        # denom = valid.sum(dim=1).clamp(min=1).to(logx.dtype)  # avoid /0

        # gm = (logx * valid).sum(dim=1) / denom  # (B,)

        # # Fill only valid positions; invalid remain 
        # out[valid] = (logx - gm[:, None])[valid]
#########

        # zero out invalid positions
        xv = torch.where(valid, x, torch.zeros_like(x))
        # add pseudocount ONLY to valid taxa
        xv = xv + pseudocount * valid
        # relative abundances
        denom = xv.sum(dim=1, keepdim=True).clamp_min(pseudocount)
        p = xv / denom   # (B, L), in [0,1]
        # arcsine–sqrt
        out[valid] = torch.asin(torch.sqrt(p[valid]))        


        # if valid.any():
        #     xv = x[valid] + pseudocount
        #     logx = torch.log(xv)
        #     gm = logx.mean()
        #     out[valid] = logx - gm  # CLR values (can be negative)

        # keep pad/class positions at 0.0
        return out