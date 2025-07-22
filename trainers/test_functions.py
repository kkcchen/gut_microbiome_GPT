import numpy as np
import os
import torch
import json

from data_utils.preprocessor import Preprocessor
from data_utils.tokenizer import Tokenizer
from trainers import logger
from data_utils.vocab import MicrobiomeVocab

from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, confusion_matrix

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
            probs, targets = accelerator.gather_for_metrics((probs, targets))
            all_probs.append(probs)
            all_targets.append(targets)
    
    all_probs = torch.cat(all_probs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    return all_probs, all_targets


def evaluate_classification(model, dataloader, batch_vocab, vocab_pad_index, output_path, accelerator):
    probs, targets = get_class_probs(model, dataloader, vocab_pad_index, accelerator)
    probs = probs.cpu().numpy()
    targets = targets.cpu().numpy()
    
    if accelerator.is_main_process:
        print("shape of probs and targets is:", probs.shape, targets.shape)
        print("location of probs and targets is", probs.device, targets.device)
        predictions = np.argmax(probs, axis=1)
        total_accuracy = accuracy_score(targets, predictions)
        conf_mat = confusion_matrix(targets, predictions)
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
        conf_row_strs = [str(row) for row in conf_mat]

        region_scores.sort(key=lambda x: x["Region"])
        region_scores.append({"Total Accuracy": total_accuracy,
                              "Categories": batch_vocab.itos,
                              "Confusion Matrix": conf_row_strs})
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(region_scores, f, indent=4)

        print(f"Scores for all regions saved to {output_path}")


def evaluate_regression(model, dataloader, vocab_pad_index, output_path, accelerator):
    """
    Evaluate the model on a regression task.
    
    Args:
        model: The model to evaluate.
        dataloader: The dataloader containing the test data.
        vocab_pad_index: The padding index for the vocabulary.
        output_path: Path to save the evaluation results.
        accelerator: The Accelerator instance for distributed training.
    
    Returns:
        None
    """
    all_preds = []
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
                preds = output_dict["logits"]
            
            preds, targets = accelerator.gather_for_metrics((preds, targets))
            all_preds.append(preds)
            all_targets.append(targets)
    
    all_preds = torch.cat(all_preds, dim=0).cpu().numpy()
    all_targets = torch.cat(all_targets, dim=0).cpu().numpy()
    
    if accelerator.is_main_process:
        mse = np.mean((all_preds - all_targets) ** 2)
        mae = np.mean(np.abs(all_preds - all_targets))
        
        results = {
            "Mean Squared Error": mse,
            "Mean Absolute Error": mae
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=4)
        
        print(f"Regression evaluation results saved to {output_path}")