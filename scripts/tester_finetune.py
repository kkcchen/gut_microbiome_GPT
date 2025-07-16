import argparse
from accelerate import Accelerator
import json
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from data_utils.vocab import BatchVocab
import numpy as np
import os
from trainers import logger

from trainers.finetune_functions import (
    load_model,
)
from trainers.test_functions import (
    restore_vocab,
    create_testdata_state,
    get_class_probs
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
    model = load_model(args.model_config_path, args.best_path)
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
        gen_percent=0.0,  # No generation for encoding
        vocab=vocab,
        batch_size=64,  # Adjust batch size as needed
        shuffle=False,  # Do not shuffle for encoding
    )
    
    # Prepare model and dataloader with accelerate
    model, dataloader = accelerator.prepare(model, dataloader)
    
    # evaluate
    probs, targets = get_class_probs(model, dataloader, vocab.pad_index, accelerator)
    probs = probs.cpu().numpy()
    targets = targets.cpu().numpy()
    
    if accelerator.is_main_process:
        print("shape of probs and targets is:", probs.shape, targets.shape)
        print("location of probs and targets is", probs.device, targets.device)
        predictions = np.argmax(probs, axis=1)
        total_accuracy = accuracy_score(targets, predictions)
        region_scores = []
        for region, index in batch_vocab.stoi.items():
            logger.info(f"region {region} is {index}")
            if region == "unknown":
                continue
            scores = probs[:, index]
            binary_predictions = np.array((predictions == index), dtype=int)
            binary_targets = np.array((targets == index), dtype=int)
            n_samples = np.sum(binary_targets).item()
            accuracy = accuracy_score(binary_targets, binary_predictions)
            auroc = roc_auc_score(binary_targets, scores)
            aupr = average_precision_score(binary_targets, scores)
            baseline_precision = np.mean(binary_targets)
            
            
            region_scores.append({
                "Region": region,
                "n_samples": n_samples,
                "Accuracy": accuracy,
                "AUC (ROC)": auroc,
                "Average Precision": aupr,
                "Baseline Precision": baseline_precision
            })
        
        # Save the scores to a file
        region_scores.sort(key=lambda x: x["Region"])
        region_scores.append({"Total Accuracy": total_accuracy})
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, "w") as f:
            json.dump(region_scores, f, indent=4)

        print(f"Scores for all regions saved to {args.output_path}")
    
    


if __name__ == "__main__":
    main()
    