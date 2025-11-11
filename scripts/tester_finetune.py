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
    evaluate_classification,
    evaluate_regression
)
from data_utils.dataloader import (
    prepare_dataloader
)

def main():
    parser = argparse.ArgumentParser(description="Train TransformerModel on microbiome data")
    parser.add_argument("--best-path", type=str, required=True, help="Path best model is saved in")
    parser.add_argument("--model-config-path", type=str, required=True, help="Path to model configuration file")

    parser.add_argument("--vocab-dir", type=str, required=True, help="Dir to the vocabulary files")
    
    parser.add_argument("--anndata-path", type=str, required=True, help="Path to test .h5ad file (samples, taxa, 2)")

    parser.add_argument("--nrows", type=int, default=None, help="Number of rows to use from the h5ad file (for debugging)")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to save the output files.")
    parser.add_argument("--task-type", type=str, choices=["classification", "regression"], required=True, help="Specify the task type: classification or regression")
    parser.add_argument("--ignored-labels", type=str, nargs='*', default=["unknown"], help="List of labels to ignore in the target column.")
    parser.add_argument("--downstream-task", type=str, required=True, help="Type of downstream task to perform.")


    args = parser.parse_args()
    is_classification = (args.task_type == "classification")
    
    # Initialize accelerator
    accelerator = Accelerator()

    # load model config and then model
    model, model_config = load_finetuned_model(args.model_config_path, args.best_path)
    model.eval()
    
    use_gnn = model_config.get("base_model_config", {}).get("use_gnn", False)
    
    batch_obskey = None
    continuous_obskey = None
    
    if is_classification:
        batch_obskey = "categorical_label"
    else:
        continuous_obskey = "continuous_label"
        train_mean = model_config["train_mean"]
        train_std = model_config["train_std"]
        
    # load data and prepare dataloader
    vocab, batch_vocab, adata, graph_data = restore_vocab_test(args.anndata_path, args.vocab_dir, downstream_task=args.downstream_task, batch_obskey=batch_obskey, use_gnn=use_gnn, nrows=args.nrows, accelerator=accelerator)
    num_bins = model.base_model.n_input_bins
    
    adata = adata[adata.obs['downstream_task'] == args.downstream_task]
    if is_classification:
        adata = adata[~adata.obs[batch_obskey].isin(args.ignored_labels)]
    else:
        adata = adata[~adata.obs[continuous_obskey].isin(args.ignored_labels)]
        
    data_dict, adata = create_testdata_state(
        adata=adata,
        num_bins=num_bins,
        vocab=vocab,
        batch_obskey=batch_obskey,
        continuous_obskey=continuous_obskey,
        nrows=args.nrows  # Set to None to use all rows
    )
    
    if not is_classification:
        data_dict["continuous_labels"] = (data_dict["continuous_labels"] - train_mean) / train_std
    
    dataloader = prepare_dataloader(
        data_dict,
        use_batch_labels=is_classification,
        use_continuous_labels=not is_classification,
        gen_percent=0.0,  # No generation for encoding
        vocab=vocab,
        batch_size=64,  # Adjust batch size as needed
        shuffle=False,  # Do not shuffle for encoding
    )
    
    # Prepare model and dataloader with accelerate
    model, dataloader = accelerator.prepare(model, dataloader)

    # evaluate
    if is_classification:
        evaluate_classification(model, dataloader, batch_vocab, vocab.pad_index, args.output_dir, accelerator, graph_data)
    else:
        evaluate_regression(model, dataloader, vocab.pad_index, train_mean, train_std, args.output_dir, accelerator, graph_data)

if __name__ == "__main__":
    main()
    