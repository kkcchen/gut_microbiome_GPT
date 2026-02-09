import numpy as np
import os
import torch
import json
import anndata as ad
from sklearn.metrics import RocCurveDisplay, r2_score
import matplotlib.pyplot as plt

from trainers import logger
import seaborn as sns

from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, confusion_matrix, f1_score


def get_class_probs(model, dataloader, vocab_pad_index, accelerator, graph_data):
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
                    graph_data=graph_data
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
        y_pred=scores,
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
    logger.info(f"\t Saved combined ROC plot to {output_path}")

def save_confusion_matrix(confusion_matrix, categories, output_dir):
    # Ensure array
    confusion_matrix = np.array(confusion_matrix, dtype=np.float64)

    # Normalized for colors
    row_sums = confusion_matrix.sum(axis=1, keepdims=True)
    confusion_matrix_normalized = confusion_matrix / row_sums

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        confusion_matrix_normalized,              # use normalized values for colors
        annot=confusion_matrix.astype(int),       # show raw counts in text
        fmt="d",
        cmap="Blues",
        xticklabels=categories,
        yticklabels=categories
    )
    
    plt.title("Confusion Matrix (counts with normalized colors)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(output_path)
    plt.close()
    logger.info(f"\t Saved confusion matrix plot to {output_path}")

def evaluate_binary(label_name, y_probs, y_pred, y_test_binary):
    
    # Check that all shapes are equal
    assert y_probs.shape == y_pred.shape == y_test_binary.shape, \
        f"\t Shape mismatch: y_probs {y_probs.shape}, y_pred {y_pred.shape}, y_test_binary {y_test_binary.shape}"
    # print(f"\t y_probs {y_probs.shape}, y_pred {y_pred.shape}, y_test_binary {y_test_binary.shape}")
    
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
    conf_mat = confusion_matrix(y_true, y_pred, labels=sorted(train_class_labels))
    label_scores = []
    fig, ax = plt.subplots(figsize=(8, 6))

    index_label_pairs = enumerate(train_class_labels)
    index_label_pairs = sorted(index_label_pairs, key=lambda x: x[1])
    for index, label in index_label_pairs:
        scores = y_probs[:, index]
        binary_predictions = (y_pred == label).astype(int)
        binary_targets = (y_true == label).astype(int)
        if binary_targets.sum() == 0:
            logger.warning(f"\t Skipping label {label} as it has no positive samples")
            continue
        label_scores.append(evaluate_binary(label, scores, binary_predictions, binary_targets))
        add_roc_curve(binary_targets, scores, label, ax)
    
    save_roc_curve(ax, output_dir)
    save_confusion_matrix(conf_mat, sorted(train_class_labels), output_dir)
    # Save the scores to a file
    conf_row_strs = [str(row) for row in conf_mat]

    label_scores.sort(key=lambda x: x["Label"])
    label_scores.append({"Total Accuracy": total_accuracy,
                        "Micro F1": micro_f1,
                        "Macro F1": macro_f1,
                        "Categories": list(sorted(train_class_labels)),
                        "Confusion Matrix": conf_row_strs})
    # Save the scores to a file
    scores_file = os.path.join(output_dir, "multiclass_scores.json")
    logger.info(f"\t Scores for all labels saved to {scores_file}")
    with open(scores_file, "w") as f:
        json.dump(label_scores, f, indent=4)

    logger.info(f"\t Scores for all labels saved to {scores_file}")

def evaluate_multiclass_perm_test(y_true, y_probs, train_class_labels):
    # evaluate, done for all
    y_true = np.array(y_true)
    train_class_labels = np.array(train_class_labels)
    y_pred_index = np.argmax(y_probs, axis=1)
    y_pred = train_class_labels[y_pred_index]
    # print("types of predictions and targets are:", y_pred.dtype, y_true.dtype)
    total_accuracy = accuracy_score(y_true, y_pred)
    micro_f1 = f1_score(y_true, y_pred, average='micro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    conf_mat = confusion_matrix(y_true, y_pred, labels=sorted(train_class_labels))
    roc_dict = {
        "Label": [],
        "n_samples": [],
        "Accuracy": [],
        "AUC (ROC)": [],
        "Average Precision": [],
        "Baseline Precision": []
    }

    index_label_pairs = enumerate(train_class_labels)
    index_label_pairs = sorted(index_label_pairs, key=lambda x: x[1])
    for index, label in index_label_pairs:
        scores = y_probs[:, index]
        binary_predictions = (y_pred == label).astype(int)
        binary_targets = (y_true == label).astype(int)
        if binary_targets.sum() == 0:
            logger.warning(f"\t Skipping label {label} as it has no positive samples")
            continue
        roc = evaluate_binary(label, scores, binary_predictions, binary_targets)
        for key in roc_dict.keys():
            roc_dict[key].append(roc[key])
    return total_accuracy, micro_f1, macro_f1, conf_mat, roc_dict


def evaluate_classification(model, dataloader, batch_vocab, vocab_pad_index, output_dir, accelerator, graph_data):
    probs, targets = get_class_probs(model, dataloader, vocab_pad_index, accelerator, graph_data)
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
    
    logger.info(f"Regression evaluation results saved to {scores_file}")

def evaluate_regression_perm_test(y_true, y_pred, mean=0, std=1):
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
    
    return float(mse), float(mae), float(r2)