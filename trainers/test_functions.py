import numpy as np
import os
import torch

from data_utils.preprocessor import Preprocessor
from data_utils.tokenizer import Tokenizer
from trainers import logger
from data_utils.vocab import MicrobiomeVocab


def restore_vocab(vocab_path, vocab_metadata_path):
    # vocab should always already exist
    if not os.path.exists(vocab_path) or not os.path.exists(vocab_metadata_path):
        raise FileNotFoundError(f"Vocab files not found at {vocab_path} or {vocab_metadata_path}")
    
    vocab = MicrobiomeVocab.get_vocab_from_json(vocab_path, vocab_metadata_path)
    logger.info(f"Vocab loaded from {vocab_path} and {vocab_metadata_path}")
    
    return vocab


def create_testdata_state(npy_path, num_bins, vocab, batch_labels=None, batch_vocab=None, nrows=None):
    # create the data dict
    if nrows:
        hmc_npy = np.load(npy_path)[:nrows,:,:] # shape (num_samples, num_taxa, 2) where (:,:,0) is taxa_id and (:,:,1) is counts
    else:
        hmc_npy = np.load(npy_path)

    preprocessor = Preprocessor(
        binning=num_bins,
    )

    _, _ = preprocessor.process_from_np(hmc_npy)

    # create tokenizer
    tokenizer = Tokenizer(vocab, batch_vocab=batch_vocab)
    data_dict = tokenizer.tokenize_and_pad_batch(hmc_npy, batch_labels=batch_labels)
    # Assuming data_dict is a dictionary with keys 'taxa_ids', 'values', and 'batch_labels' if added
    
    if batch_labels:
        # remove rows where batch_labels are not in the vocab (marked as -1)
        valid_indices = data_dict["batch_labels"] != -1
        data_dict["taxa_ids"] = data_dict["taxa_ids"][valid_indices]
        data_dict["values"] = data_dict["values"][valid_indices]
        data_dict["batch_labels"] = data_dict["batch_labels"][valid_indices]

    return data_dict


def get_class_probs(model, dataloader, vocab_pad_index, accelerator):
    """
    Get class probabilities from the model for the given dataloader.
    
    Args:
        model: The model to evaluate.
        dataloader: The dataloader containing the test data.
        accelerator: The Accelerator instance for distributed training.
    
    Returns:
        A list of class probabilities for each batch in the dataloader.
    """
    all_probs = []
    all_targets = []
    model.eval()
    
    with torch.no_grad():
        for data_dict in dataloader:
            taxa = data_dict["ids"]
            values = data_dict["values"]
            key_padding_mask = taxa.eq(vocab_pad_index)
            targets = data_dict["batch_labels"]

            with accelerator.autocast():
                output_dict = model(
                    taxa,
                    values,
                    src_key_padding_mask=key_padding_mask,
                )
                class_logits = output_dict["logits"]
            # Convert logits to probabilities
            probs = torch.softmax(class_logits, dim=-1)
            gathered_probs = accelerator.gather_for_metrics(probs)
            gathered_targets = accelerator.gather_for_metrics(targets)
            
            all_probs.append(gathered_probs.cpu())
            all_targets.append(gathered_targets.cpu())
    
    all_probs = torch.cat(all_probs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    return all_probs, all_targets
