import argparse
from accelerate import Accelerator
import json
import numpy as np
import os
from trainers import logger

from trainers.finetune_functions import (
    load_finetuned_model,
)
from trainers.test_functions import (
    restore_vocab_test,
    create_testdata_state,
    evaluate_classification
)
from data_utils.dataloader import (
    prepare_dataloader
)

def main():
    parser = argparse.ArgumentParser(description="Train TransformerModel on microbiome data")
    parser.add_argument("--best-path", type=str, required=True, help="Path best model is saved in")
    parser.add_argument("--model-config-path", type=str, required=True, help="Path to model configuration file")

    parser.add_argument("--finetune-vocab-path", type=str, required=True, help="Path to the vocabulary JSON file")
    
    parser.add_argument("--anndata-path", type=str, required=True, help="Path to test .h5ad file (samples, taxa, 2)")

    parser.add_argument("--nrows", type=int, default=None, help="Number of rows to use from the h5ad file (for debugging)")
    parser.add_argument("--output-path", type=str, required=True, help="Path to save the output files.")
    
    args = parser.parse_args()
    
    # Initialize accelerator
    accelerator = Accelerator()

    # load model config and then model
    model = load_finetuned_model(args.model_config_path, args.best_path)
    model.eval()
        
    batch_obskey = "location"
    
    # load data and prepare dataloader
    vocab, batch_vocab, adata = restore_vocab_test(args.anndata_path, args.finetune_vocab_dir, batch_obskey=batch_obskey)
    num_bins = model.base_model.n_input_bins
    
    data_dict = create_testdata_state(
        adata=adata,
        num_bins=num_bins,
        vocab=vocab,
        batch_obskey=batch_obskey,
        nrows=args.nrows  # Set to None to use all rows
    )
    
    dataloader = prepare_dataloader(
        data_dict,
        use_batch_labels=True,
        use_continuous_labels=False,
        gen_percent=0.0,  # No generation for encoding
        vocab=vocab,
        batch_size=64,  # Adjust batch size as needed
        shuffle=False,  # Do not shuffle for encoding
    )
    
    # Prepare model and dataloader with accelerate
    model, dataloader = accelerator.prepare(model, dataloader)

    # evaluate
    evaluate_classification(model, dataloader, batch_vocab, vocab.pad_index, args.output_path, accelerator)

if __name__ == "__main__":
    main()
    