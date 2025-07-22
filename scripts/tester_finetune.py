import argparse
from accelerate import Accelerator
import json
from data_utils.vocab import BatchVocab
import numpy as np
import os
from trainers import logger

from trainers.finetune_functions import (
    load_finetuned_model,
)
from trainers.test_functions import (
    restore_vocab,
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

    parser.add_argument("--batch-vocab-path", type=str, required=True, help="Path to the batch vocabulary JSON file")

    parser.add_argument("--vocab-path", type=str, required=True, help="Path to the vocabulary JSON file")
    parser.add_argument("--vocab-metadata-path", type=str, required=True, help="Path to the vocabulary metadata JSON file")
    
    parser.add_argument("--npy-path", type=str, required=True, help="Path to test .npy file (samples, taxa, 2)")
    parser.add_argument("--test-loc-labels-path", type=str, required=True, help="Path to the test location labels file.")

    parser.add_argument("--nrows", type=int, default=None, help="Number of rows to use from the npy file (for debugging)")
    parser.add_argument("--output-path", type=str, required=True, help="Path to save the output files.")
    
    args = parser.parse_args()
    
    # Initialize accelerator
    accelerator = Accelerator()

    # load model config and then model
    model = load_finetuned_model(args.model_config_path, args.best_path)
    model.eval()
    
    # read test location labels
    with open(args.test_loc_labels_path, 'r') as f:
        loc_labels = json.load(f)
        
    if args.nrows:
        loc_labels = loc_labels[:args.nrows]
    
    # convert loc labels to indices according to known classes
    batch_vocab = BatchVocab.get_vocab_from_json(args.batch_vocab_path)
    
    # load data and prepare dataloader
    vocab = restore_vocab(args.vocab_path, args.vocab_metadata_path)
    num_bins = model.base_model.n_input_bins
    
    data_dict = create_testdata_state(
        npy_path=args.npy_path,
        num_bins=num_bins,
        vocab=vocab,
        batch_labels=loc_labels,
        batch_vocab=batch_vocab,
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
    