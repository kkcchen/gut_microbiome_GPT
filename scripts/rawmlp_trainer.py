import numpy as np
import time
import os
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, confusion_matrix, f1_score, mean_squared_error

import anndata as ad
import matplotlib.pyplot as plt

from models import RawMLP
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
from sklearn.preprocessing import LabelEncoder

import wandb

import argparse

from trainers.test_functions import evaluate_multiclass_and_save, evaluate_regression_and_save
from transformers import get_cosine_schedule_with_warmup

def train_one_epoch(model, loader, optimizer, loss_fn, device, is_classification):
    model.train()
    total_loss = 0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)

        optimizer.zero_grad()
        output = model(Xb)["logits"]
        if not is_classification:
            output = output.squeeze(-1)

        loss = loss_fn(output, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, loss_fn, device, is_classification):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            output = model(Xb)["logits"]
            if not is_classification:
                output = output.squeeze(-1)

            loss = loss_fn(output, yb)
            total_loss += loss.item()

            if is_classification:
                preds = output.argmax(dim=1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(yb.cpu().numpy())
            else:
                all_preds.append(output.cpu().numpy())
                all_labels.append(yb.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    avg_loss = total_loss / len(loader)

    if is_classification:
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="weighted")
        return avg_loss, acc, f1
    else:
        mse = mean_squared_error(all_labels, all_preds)
        return avg_loss, mse, None


def main():
    parser = argparse.ArgumentParser(description="Script for processing embeddings.")
    parser.add_argument("--train-embed-path", type=str, required=True, help="Path to the anndata where the train embeddings are saved or should be saved in the obsm['embedding'].")
    parser.add_argument("--test-embed-path", type=str, required=True, help="Path to the anndata where the test embeddings are saved or should be saved in the obsm['embedding'].")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the best model and parameters.")
    parser.add_argument("--model-type", type=str, choices=["classification", "regression"], required=True, help="Type of model to train: 'multiclass' for one model handling all classes, 'one-vs-all' for one model per class, or 'regression' for regression tasks.")
    parser.add_argument("--target-colname", type=str, required=True, help="Column name in the anndata obs to use as target labels.")
    parser.add_argument("--emb-name", type=str, default="raw_embedding", help="Name of the obsm key where embeddings are stored.")
    parser.add_argument("--ignored-labels", type=str, nargs='*', default=["unknown"], help="List of labels to ignore in the target column.")
    parser.add_argument("--downstream-task", type=str, required=True, help="Type of downstream task to perform.")
    
    # wandb
    parser.add_argument("--wandb-enabled", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-entity", type=str, default=None, help="wandb entity name")
    parser.add_argument("--wandb-project", type=str, default=None, help="wandb project name")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="wandb run name")
    parser.add_argument("--wandb-run-notes", type=str, default=None, help="wandb run notes")

    args = parser.parse_args()

    train_embed_path = args.train_embed_path
    test_embed_path = args.test_embed_path
    output_dir = args.output_dir
    
    emb_name = args.emb_name
    is_classification = args.model_type == "classification"
        
    print("args are:", args)
    
    # Load train embeddings and labels
    train_adata = ad.read_h5ad(train_embed_path)
    if args.downstream_task == "location":
        assert "unknown" in args.ignored_labels, "For location task, 'unknown' must be in ignored labels."
    train_adata = train_adata[train_adata.obs['downstream_task'] == args.downstream_task]
    train_adata = train_adata[~train_adata.obs[args.target_colname].isin(args.ignored_labels)]
    X_train = train_adata.obsm[emb_name]
    Y_train = train_adata.obs[args.target_colname]
    
    print("X_train shape:", X_train.shape)
    print("Y_train shape:", Y_train.shape)

    # Load test embeddings and labels
    test_adata = ad.read_h5ad(test_embed_path)
    test_adata = test_adata[test_adata.obs['downstream_task'] == args.downstream_task]        
    test_adata = test_adata[~test_adata.obs[args.target_colname].isin(args.ignored_labels)]
    X_test = test_adata.obsm[emb_name]
    Y_test = test_adata.obs[args.target_colname]    
    
    print("X_test shape:", X_test.shape)
    print("Y_test shape:", Y_test.shape)
    if is_classification:
        le = LabelEncoder()
        Y_train_enc = le.fit_transform(Y_train)
        Y_test_enc = le.transform(Y_test)
        n_cls = len(le.classes_)
    else:
        Y_train_enc = Y_train.astype(np.float32)
        Y_test_enc = Y_test.astype(np.float32)

        # Standardize the target values
        mean = Y_train_enc.mean()
        std = Y_train_enc.std()
        Y_train_enc = (Y_train_enc - mean) / std
        Y_test_enc = (Y_test_enc - mean) / std
        n_cls = 1

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train_enc, dtype=torch.long if is_classification else torch.float32)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test_enc, dtype=torch.long if is_classification else torch.float32)

    train_ds = TensorDataset(X_train_tensor, Y_train_tensor)
    test_ds = TensorDataset(X_test_tensor, Y_test_tensor)
    
    train_ds, val_ds = torch.utils.data.random_split(train_ds, [int(0.8 * len(train_ds)), len(train_ds) - int(0.8 * len(train_ds))])

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
    
    print("about to start training or loading models")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RawMLP(
        input_dim=X_train.shape[1],
        hidden_dim=256,
        n_cls=n_cls,
        nlayers=3,
    ).to(device)

    if is_classification:
        loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    num_training_steps = len(train_loader) * 20  # Assuming 20 epochs
    num_warmup_steps = int(0.1 * num_training_steps)  # 10% of training steps for warmup
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    
    if args.wandb_enabled:
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.wandb_run_name,
            notes=args.wandb_run_notes,
            config={
                "model_type": args.model_type,
                "downstream_task": args.downstream_task,
                "target_colname": args.target_colname,
                "emb_name": args.emb_name,
                "ignored_labels": args.ignored_labels,
                "train_embed_path": args.train_embed_path,
                "test_embed_path": args.test_embed_path,
                "batch_size": 64,
                "learning_rate": 1e-3,
                "hidden_dim": 256,
                "nlayers": 3,
                "n_epochs": 20
            }
        )
        wandb.watch(model, log="all")

    n_epochs = 20
    best_val_loss = float("inf")

    for epoch in range(n_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, is_classification)
        val_loss, val_metric, val_extra = evaluate(model, val_loader, loss_fn, device, is_classification)

        if is_classification:
            print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_metric:.4f} | Val F1: {val_extra:.4f}")
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "val/accuracy": val_metric,
                "val/f1": val_extra
            })
        else:
            print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | MSE: {val_metric:.4f}")
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "val/mse": val_metric
            })
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
            print(f"Best model saved at epoch {epoch+1}")

        scheduler.step()
        
    
    # Final evaluation on test set
    with torch.no_grad():
        model.load_state_dict(torch.load(os.path.join(output_dir, "best_model.pth")))
        model.eval()
        y_logits = []
        y_true = []
        
        for Xb, yb in test_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            logits = model(Xb)["logits"]
            y_logits.append(logits.cpu().numpy())
            y_true.append(yb.cpu().numpy())

        y_logits = np.concatenate(y_logits)
        y_true = np.concatenate(y_true)
        
        if is_classification:
            y_probs = torch.softmax(torch.tensor(y_logits), dim=-1).numpy()
            y_true_str = le.inverse_transform(y_true)
            evaluate_multiclass_and_save(
                y_true=y_true_str,
                y_probs=y_probs,
                train_class_labels=le.classes_,
                output_dir=output_dir
            )
        else:
            assert y_logits.shape[1] == 1, f"Second dimension of predictions should be 1, got {y_logits.shape}"
            evaluate_regression_and_save(
                y_true=y_true,
                y_pred=y_logits.squeeze(-1),
                output_dir=output_dir,
                mean=mean,
                std=std
            )
    
    
        
if __name__ == "__main__":
    print("Starting mlp experiment script")
    main()
