import numpy as np
import os
import torch
import json
import anndata as ad

from data_utils.preprocessor import Preprocessor
from data_utils.tokenizer import Tokenizer
from trainers import logger
from data_utils.vocab import MicrobiomeVocab, BatchVocab


from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, confusion_matrix

def restore_vocab_test(anndata_path, vocab_restore_path, batch_obskey=None):
    if os.path.exists(vocab_restore_path):
        # load the vocab from the file
        adata_vocab = ad.read_h5ad(vocab_restore_path)
        assert "vocab_metadata" in adata_vocab.uns and "taxa_id" in adata_vocab.var, "vocab metadata or taxa_id not found in the adata"
        vocab = MicrobiomeVocab.restore_vocab(adata_vocab)
        
        if batch_obskey is not None:
            batch_vocab = BatchVocab.restore_batchvocab(adata_vocab, batch_obskey)
        else:
            batch_vocab = None
        
        logger.info(f"Vocab and/or batch vocab restored from {vocab_restore_path}")
    else:
        raise FileNotFoundError(f"Vocab file not found at {vocab_restore_path}")

    adata = ad.read_h5ad(anndata_path)
    adata.var["taxa_id"] = adata.var["taxa"].map(vocab.stoi)
    
    if batch_obskey:
        batch_obskey_id = f"{batch_obskey}_id"
        adata.obs[batch_obskey_id] = adata.obs[batch_obskey].map(batch_vocab.stoi)
        # remove samples with NaN in batch_obskey_id
        adata = adata[~adata.obs[batch_obskey_id].isna(), :].copy()
    
    return vocab, batch_vocab, adata


def create_testdata_state(adata, num_bins, vocab, batch_obskey, nrows=None):
    # create the data dict
    if nrows:
        adata = adata[:nrows, :].copy()
    
    preprocessor = Preprocessor(
        binning=num_bins,
    )
    
    hmc_npy = np.array(adata.layers["top_512"].todense(), dtype=np.float32)
    taxa_ids = np.array(adata.var["taxa_id"])
    
    stacked_rows, _ = preprocessor.process_from_np(hmc_npy, taxa_ids)

    # create tokenizer
    adata.layers["binned_rows"] = stacked_rows

    tokenizer = Tokenizer(vocab)
    data_dict = tokenizer.tokenize_and_pad_batch(adata, batch_obskey=batch_obskey)
    
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