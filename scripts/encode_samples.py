import torch
import argparse
import os
import numpy as np
import json

from safetensors.torch import load_file
from data_utils.preprocessor import Preprocessor
from data_utils.dataloader import prepare_dataloader
from data_utils.tokenizer import Tokenizer

from accelerate import Accelerator
from trainers import logger

from data_utils.vocab import MicrobiomeVocab

def restore_vocab(vocab_path, vocab_metadata_path):
    # vocab should always already exist
    if not os.path.exists(vocab_path) or not os.path.exists(vocab_metadata_path):
        raise FileNotFoundError(f"Vocab files not found at {vocab_path} or {vocab_metadata_path}")
    
    vocab = MicrobiomeVocab.get_vocab_from_json(vocab_path, vocab_metadata_path)
    logger.info(f"Vocab loaded from {vocab_path} and {vocab_metadata_path}")
    
    return vocab


def create_or_restore_testdata_state(npy_path, num_bins, vocab, accelerator: Accelerator, nrows=None):
    # check if the data state already exists. if not, create the data dict
    if nrows:
        hmc_npy = np.load(npy_path)[:nrows,:,:] # shape (num_samples, num_taxa, 2) where (:,:,0) is taxa_id and (:,:,1) is counts
    else:
        hmc_npy = np.load(npy_path)

    preprocessor = Preprocessor(
        binning=num_bins,
    )

    _, _ = preprocessor.process_from_np(hmc_npy)

    # create tokenizer
    tokenizer = Tokenizer(vocab)
    data_dict = tokenizer.tokenize_and_pad_batch(hmc_npy)
    # Assuming data_dict is a dictionary with keys 'taxa_ids', 'values'

    return data_dict


def main():
    parser = argparse.ArgumentParser(description="Encode data with a model using safetensors weights.")
    parser.add_argument("--safetensors-path", type=str, required=True, help="Path to the .safetensors file")
    parser.add_argument("--output-path", type=str, required=True, help="Where to save the output tensor")
    parser.add_argument("--vocab-path", type=str, required=True, help="Path to the vocabulary JSON file")
    parser.add_argument("--vocab-metadata-path", type=str, required=True, help="Path to the vocabulary metadata JSON file")
    parser.add_argument("--model-config-path", type=str, required=True, help="Path to the model configuration file (not used in this script but can be useful for reference)")
    parser.add_argument("--npy-path", type=str, required=True, help="Path to the training data numpy file")
    parser.add_argument("--nrows", type=int, default=None, help="Number of rows to use from the npy file (for debugging)")
    args = parser.parse_args()
    
    safetensors_path = args.safetensors_path
    output_path = args.output_path
    vocab_path = args.vocab_path
    vocab_metadata_path = args.vocab_metadata_path
    npy_path = args.npy_path
    model_config_path = args.model_config_path
    
    nrows = args.nrows
    
    # Initialize accelerator
    accelerator = Accelerator()
    
    # restore vocab
    vocab = restore_vocab(vocab_path, vocab_metadata_path)
    
    with open(model_config_path, 'r') as f:
        model_config = json.load(f)

    # === Load model ===
    from models import TransformerModel
    model = TransformerModel(**model_config)

    state_dict = load_file(safetensors_path)
    model.load_state_dict(state_dict)
    model.eval()

    # === Load test dataloader ===
    data_dict = create_or_restore_testdata_state(
        npy_path=npy_path,
        num_bins=model_config["n_input_bins"],
        vocab=vocab,
        accelerator=accelerator,
        nrows=nrows  # Set to None to use all rows
    )
    
    dataloader = prepare_dataloader(
        data_dict,
        use_batch_labels=False,
        gen_percent=0.0,  # No generation for encoding
        vocab=vocab,
        batch_size=64,  # Adjust batch size as needed
        shuffle=False,  # Do not shuffle for encoding
    )

    # Prepare model and dataloader with accelerate
    model, dataloader = accelerator.prepare(model, dataloader)

    # === Encode ===
    all_cell_embs = []
    with torch.no_grad():
        for data_dict in dataloader:
            taxa = data_dict["ids"]
            values = data_dict["values"]
            src_padding_mask = taxa.eq(vocab.pad_index)
            unwrapped_model = accelerator.unwrap_model(model)
            output, _ = unwrapped_model.encode(
                src=taxa,  # (batch, seq_len)
                values=values,  # (batch, seq_len)
                src_key_padding_mask=src_padding_mask,  # (batch, seq_len)
            )  # (batch, seq_len, embsize)
            cell_emb = unwrapped_model.get_cell_emb_from_layer(output)  # (batch, embsize)
            gathered = accelerator.gather_for_metrics(cell_emb)  # Gather across processes
            all_cell_embs.append(gathered.cpu())

    # === Save result (only main process) ===
    if accelerator.is_main_process:
        final_tensor = torch.cat(all_cell_embs, dim=0)
        print(f"number of samples is {final_tensor.shape[0]}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, final_tensor.cpu().numpy())
        print(f"Saved cell embeddings as NumPy array to {output_path}")

if __name__ == "__main__":
    main()
