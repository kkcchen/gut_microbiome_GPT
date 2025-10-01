import numpy as np
import os
import torch
import json
import anndata as ad
from sklearn.metrics import RocCurveDisplay, r2_score
import matplotlib.pyplot as plt

from data_utils.preprocessor import Preprocessor
from data_utils.tokenizer import Tokenizer
from trainers import logger
from data_utils.vocab import MicrobiomeVocab, BatchVocab


from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, confusion_matrix, f1_score

def restore_vocab_test(anndata_path, vocab_restore_path, batch_obskey=None):
    if os.path.exists(vocab_restore_path):
        # load the vocab from the file
        adata_vocab = ad.read_h5ad(vocab_restore_path)
        assert "vocab_metadata" in adata_vocab.uns and "taxa_id" in adata_vocab.var, "vocab metadata or taxa_id not found in the adata"
        logger.info(adata_vocab)
        vocab = MicrobiomeVocab.restore_vocab(adata_vocab)
        
        if batch_obskey is not None:
            batch_vocab = BatchVocab.restore_batchvocab(adata_vocab, batch_obskey)
        else:
            batch_vocab = None
        
        logger.info(f"Vocab and/or batch vocab restored from {vocab_restore_path}")
    else:
        raise FileNotFoundError(f"Vocab file not found at {vocab_restore_path}")

    adata = ad.read_h5ad(anndata_path)
    adata.var["taxa"] = adata.var_names
    adata.var["taxa_id"] = adata.var["taxa"].map(vocab.stoi)
    
    if batch_obskey:
        batch_obskey_id = f"{batch_obskey}_id"
        adata.obs[batch_obskey_id] = adata.obs[batch_obskey].map(batch_vocab.stoi)
        # remove samples with NaN in batch_obskey_id
        adata = adata[~adata.obs[batch_obskey_id].isna(), :].copy()
    
    # Sanity check for vocab
    # logger.info(f"First 5 elements of vocab.itos: {vocab.itos[:5]}")
    # logger.info(f"Last 3 elements of vocab.itos: {vocab.itos[-3:]}")
    # logger.info(f"Length of vocab.itos: {len(vocab.itos)}")
    logger.info("before:" + str(adata))
    return vocab, batch_vocab, adata


def create_testdata_state(adata, num_bins, vocab, batch_obskey, continuous_obskey, nrows=None):
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
    data_dict = tokenizer.tokenize_and_pad_batch(adata, batch_obskey=batch_obskey, continuous_obskey=continuous_obskey)
    
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


def add_roc_curve(binary_targets, scores, label, ax):
    RocCurveDisplay.from_predictions(
        y_true=binary_targets,
        y_score=scores,
        name=f"{label} ({binary_targets.sum()} samples)",
        plot_chance_level=False,  # Plot only once outside the loop
        ax=ax
    )


def save_roc_curve(ax, output_dir):
    # add chance line
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Chance Level')
    
    ax.set_title("ROC Curves")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.grid(True)
    ax.legend()
    
    output_path = os.path.join(output_dir, "roc_all_labels.png")
    plt.savefig(output_path)
    plt.close()
    print(f"\t Saved combined ROC plot to {output_path}")
    

def evaluate_binary(label_name, y_probs, y_pred, y_test_binary):
    
    # Check that all shapes are equal
    assert y_probs.shape == y_pred.shape == y_test_binary.shape, \
        f"\t Shape mismatch: y_probs {y_probs.shape}, y_pred {y_pred.shape}, y_test_binary {y_test_binary.shape}"
    print(f"\t y_probs {y_probs.shape}, y_pred {y_pred.shape}, y_test_binary {y_test_binary.shape}")
    
    # check that the shape of y_probs 1 dimensional
    assert y_probs.ndim == 1, f"y_probs should be 1-dimensional, got {y_probs.ndim} dimensions"
    
    # Evaluate the model's accuracy on the test set
    n_samples = np.sum(y_test_binary).item()
    accuracy = accuracy_score(y_test_binary, y_pred)
    auc = roc_auc_score(y_test_binary, y_probs)
    average_precision = average_precision_score(y_test_binary, y_probs)
    baseline_precision = np.mean(y_test_binary)

    return {
        "Label": label_name,
        "n_samples": n_samples,
        "Accuracy": accuracy,
        "AUC (ROC)": auc,
        "Average Precision": average_precision,
        "Baseline Precision": baseline_precision
    }

def evaluate_multiclass_and_save(y_true, y_probs, train_class_labels, output_dir):
    # evaluate, done for all
    os.makedirs(output_dir, exist_ok=True)
    y_true = np.array(y_true)
    train_class_labels = np.array(train_class_labels)
    y_pred_index = np.argmax(y_probs, axis=1)
    y_pred = train_class_labels[y_pred_index]
    # print("types of predictions and targets are:", y_pred.dtype, y_true.dtype)
    total_accuracy = accuracy_score(y_true, y_pred)
    micro_f1 = f1_score(y_true, y_pred, average='micro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    conf_mat = confusion_matrix(y_true, y_pred) # automatically sorts labels
    region_scores = []
    fig, ax = plt.subplots(figsize=(8, 6))

    index_label_pairs = enumerate(train_class_labels)
    index_label_pairs = sorted(index_label_pairs, key=lambda x: x[1])
    for index, label in index_label_pairs:
        scores = y_probs[:, index]
        binary_predictions = (y_pred == label).astype(int)
        binary_targets = (y_true == label).astype(int)
        if binary_targets.sum() == 0:
            print(f"\t Skipping label {label} as it has no positive samples")
            continue
        label_scores.append(evaluate_binary(label, scores, binary_predictions, binary_targets))
        add_roc_curve(binary_targets, scores, label, ax)
    
    save_roc_curve(ax, output_dir)
    # Save the scores to a file
    conf_row_strs = [str(row) for row in conf_mat]

    region_scores.sort(key=lambda x: x["Region"])
    region_scores.append({"Total Accuracy": total_accuracy,
                        "Micro F1": micro_f1,
                        "Macro F1": macro_f1,
                        "Categories": sorted(list(train_class_labels)),
                        "Confusion Matrix": conf_row_strs})
    # Save the scores to a file
    scores_file = os.path.join(output_dir, "multiclass_scores.json")
    print(f"\t Scores for all labels saved to {scores_file}")
    with open(scores_file, "w") as f:
        json.dump(label_scores, f, indent=4)

    print(f"\t Scores for all labels saved to {scores_file}")


def evaluate_classification(model, dataloader, batch_vocab, vocab_pad_index, output_dir, accelerator):
    probs, targets = get_class_probs(model, dataloader, vocab_pad_index, accelerator)
    probs = probs.cpu().numpy()
    targets = targets.cpu().numpy()
    
    train_class_labels = batch_vocab.itos
    target_labels = [batch_vocab.itos[target] for target in targets]
    if accelerator.is_main_process:
        evaluate_multiclass_and_save(
            target_labels,
            probs,
            train_class_labels,
            output_dir
        )


def plot_regression_results(y_true, y_pred, r2, output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Predicted vs. Actual
    ax1.scatter(y_true, y_pred, alpha=0.5)
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', label="Ideal")
    ax1.set_xlabel("Actual")
    ax1.set_ylabel("Predicted")
    ax1.set_title(f"Predicted vs Actual\nR² = {r2:.3f}")
    ax1.legend()
    ax1.grid(True)

    # 2. Residuals histogram
    residuals = y_pred - y_true
    ax2.hist(residuals, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax2.set_xlabel("Residual (Predicted - Actual)")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Residuals Distribution")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "regression_plots.png"))
    plt.close()


def evaluate_regression_and_save(y_true, y_pred, output_dir, mean=0, std=1):
    """
    y_true, y_pred: standardized values
    mean, std: scalars for unnormalization
    """
    # Unstandardize
    y_true_orig = y_true * std + mean
    y_pred_orig = y_pred * std + mean
    
    mse = np.mean((y_true_orig - y_pred_orig) ** 2)
    mae = np.mean(np.abs(y_true_orig - y_pred_orig))
    r2 = r2_score(y_true_orig, y_pred_orig)
    
    results = {
        "Mean Squared Error": float(mse),
        "Mean Absolute Error": float(mae),
        "R-squared": float(r2),
    }
    
    os.makedirs(output_dir, exist_ok=True)
    plot_regression_results(y_true_orig, y_pred_orig, r2, output_dir)
    scores_file = os.path.join(output_dir, "regression_scores.json")
    with open(scores_file, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"Regression evaluation results saved to {scores_file}")


def evaluate_regression(model, dataloader, vocab_pad_index, standardize_mean, standardize_std, output_dir, accelerator):
    """
    Evaluate the model on a regression task.
    
    Args:
        model: The model to evaluate.
        dataloader: The dataloader containing the test data.
        vocab_pad_index: The padding index for the vocabulary.
        output_dir: Dir to save the evaluation results.
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
            targets = data_dict["continuous_labels"]

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
    
    assert all_preds.shape[1] == 1, f"Second dimension of predictions should be 1, got {all_preds.shape[1]}"
    all_preds = all_preds.squeeze(axis=1)
    assert all_preds.shape == all_targets.shape, f"Predictions shape {all_preds.shape} does not match targets shape {all_targets.shape}"
    
    if accelerator.is_main_process:
        evaluate_regression_and_save(
            all_targets,
            all_preds,
            output_dir,
            standardize_mean,
            standardize_std
        )