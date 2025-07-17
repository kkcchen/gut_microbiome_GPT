from torch.utils.data import DataLoader
import torch

import os
import shutil
import numpy as np
import time
from safetensors.torch import load_file

from data_utils.dataloader import prepare_dataloader

from trainers.train_functions import (
    create_or_restore_data_state, epoch_end_logs
)
from trainers.finetune_functions import (
    finetune, create_training_state_finetune, init_wandb
)

from trainers.test_functions import (
    get_class_probs
)

from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score

from trainers import logger

from accelerate import Accelerator
from accelerate.utils import set_seed
import json

import argparse


def check_vocab_basemodel_match(
    base_model_config: dict,
    vocab,
) -> None:
    """
    Check if the vocab length matches the base model config.
    """
    if base_model_config["vocab_len"] != len(vocab):
        raise ValueError(
            f"vocab mismatch: base model config vocab length {base_model_config['vocab_len']} does not match the restored vocab length {len(vocab)}."
        )
    elif base_model_config["vocab_pad_index"] != vocab.pad_index:
        raise ValueError(
            f"vocab mismatch: base model config vocab pad index {base_model_config['vocab_pad_index']} does not match the restored vocab pad index {vocab.pad_index}."
        )
    elif base_model_config["vocab_pad_value"] != vocab.pad_value:
        raise ValueError(
            f"vocab mismatch: base model config vocab pad value {base_model_config['vocab_pad_value']} does not match the restored vocab pad value {vocab.pad_value}."
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train TransformerModel on microbiome data")
    parser.add_argument("--base-model-config-path", type=str, required=True, help="Path to base model configuration file")
    parser.add_argument("--base-model-path", type=str, default=None, help="Path to the base model state dictionary")
    parser.add_argument("--train-input", type=str, required=True, help="Path to train .npy file (samples, taxa, 2)")
    parser.add_argument("--best-dir", type=str, required=True, help="Directory to save best model so far")
    parser.add_argument("--data-restore-dir", type=str, required=True, help="Directory to restore data state")
    parser.add_argument("--vocab-restore-dir", type=str, required=True, help="Directory to restore vocab state")
    parser.add_argument("--batch-restore-dir", type=str, required=True, help="Directory to restore batch vocab state")
    # wandb
    parser.add_argument("--wandb-enabled", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-entity", type=str, default=None, help="wandb entity name")
    parser.add_argument("--wandb-project", type=str, default=None, help="wandb project name")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="wandb run name")

    # optional training arguments
    parser.add_argument("--init-lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--frozen-max-epochs", type=int, default=25, help="Maximum number of epochs")
    parser.add_argument("--unfrozen-max-epochs", type=int, default=50, help="Maximum number of epochs for finetuning")
    parser.add_argument("--cosine-warmup-ratio-or-step", type=float, default=0.1, help="Scheduler warmup ratio or step")
    parser.add_argument("--log-interval", type=int, default=10, help="Interval for logging")
    parser.add_argument("--patience", type=int, default=None, help="Patience for early stopping")
    parser.add_argument("--grad-accumulation-steps", type=int, default=1, help="Number of gradient accumulation steps")
    parser.add_argument("--enable-fp16", action="store_true", help="Enable mixed precision training (FP16)")

    # for debugging
    parser.add_argument("--nrows", type=int, default=None, help="For debugging to limit number of samples in set")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--start-over", action="store_true", help="Start over from scratch, ignoring existing data and checkpoints")
    parser.add_argument("--notes", type=str, default="", help="Notes for the current training run")
    
    # for finetuning
    parser.add_argument("--train-loc-labels-path", type=str, required=True, help="Path to the train location labels file.")
    parser.add_argument("--model-config-path", type=str, required=True, help="Path to save model configuration file")

    args = parser.parse_args()

    base_model_config_path = args.base_model_config_path
    base_model_path = args.base_model_path
    train_input = args.train_input
    best_dir = args.best_dir
    data_restore_dir = args.data_restore_dir
    vocab_restore_dir = args.vocab_restore_dir
    batch_restore_dir = args.batch_restore_dir

    wandb_enabled = args.wandb_enabled
    wandb_entity = args.wandb_entity
    wandb_project = args.wandb_project
    wandb_run_name = args.wandb_run_name

    init_lr = args.init_lr
    batch_size = args.batch_size
    frozen_max_epochs = args.frozen_max_epochs
    unfrozen_max_epochs = args.unfrozen_max_epochs
    cosine_warmup_ratio_or_step = args.cosine_warmup_ratio_or_step
    log_interval = args.log_interval
    patience = args.patience if args.patience else unfrozen_max_epochs + frozen_max_epochs
    grad_accumulation_steps = args.grad_accumulation_steps
    enable_fp16 = args.enable_fp16
    nrows = args.nrows
    
    train_loc_labels_path = args.train_loc_labels_path
    model_config_path = args.model_config_path
    
    start_over = args.start_over
        
    accelerator = Accelerator(gradient_accumulation_steps=grad_accumulation_steps, mixed_precision="fp16" if enable_fp16 else "no", log_with="wandb" if wandb_enabled else None)
        
    # Set random seed for reproducibility
    set_seed(args.seed, device_specific=True)
    
    with open(base_model_config_path, "r") as f:
        base_model_config = json.load(f)
        
    num_bins = base_model_config["n_input_bins"]


    # Save notes to a markdown file in the save directory
    if accelerator.is_main_process and args.notes:
        notes_path = os.path.join(best_dir, "training_notes.md")
        os.makedirs(best_dir, exist_ok=True)
        with open(notes_path, "w") as notes_file:
            notes_file.write(args.notes)

    if start_over and accelerator.is_main_process:
        logger.info("Starting over from scratch, deleting existing training state.")
        if os.path.exists(data_restore_dir):
            # The above code is using the `shutil.rmtree()` function in Python to recursively remove a
            # directory and all its contents. In this case, it is removing the directory specified by
            # the variable `data_restore_dir`.
            shutil.rmtree(data_restore_dir)
        os.makedirs(data_restore_dir, exist_ok=True)
        if os.path.exists(batch_restore_dir):
            shutil.rmtree(batch_restore_dir)
        os.makedirs(batch_restore_dir, exist_ok=True)
        # don't remove vocab_restore_dir, as it is from pretraining
    
    accelerator.wait_for_everyone()

    # Create or restore data state
    train_data_dict, valid_data_dict, vocab, batch_vocab = create_or_restore_data_state(
        train_input, num_bins, data_restore_dir, vocab_restore_dir, batch_restore_dir, accelerator, taxa_path=None, use_batch_labels=True, direct_batch_path=train_loc_labels_path, nrows=nrows, seed=args.seed
    )
    
    print(f"rank {accelerator.process_index} has vocab {batch_vocab.itos}")
    
    check_vocab_basemodel_match(base_model_config, vocab)

    class_counts = torch.bincount(train_data_dict["batch_labels"], minlength=len(batch_vocab))
    class_weights = 1.0 / (class_counts.float() + 1e-8)
    class_weights = (class_weights / class_weights.sum() * len(class_weights)).to(accelerator.device)
    loss_fn=torch.nn.CrossEntropyLoss(weight=class_weights)
    
    wandb_config={
        "learning_rate": init_lr,
        "batch_size": batch_size,
        "frozen_max_epochs": frozen_max_epochs,
        "unfrozen_max_epochs": unfrozen_max_epochs,
        "cosine_warmup_ratio_or_step": cosine_warmup_ratio_or_step,
        "binning": num_bins
    }

    logger.info("Preparing dataloaders...")
    train_loader = prepare_dataloader(
        train_data_dict,
        use_batch_labels=True,
        vocab=vocab,
        batch_size=batch_size,
        shuffle=True,
        gen_percent=0,
        contrastive_embedding=False,
    )
    valid_loader = prepare_dataloader(
        valid_data_dict,
        use_batch_labels=True,
        vocab=vocab,
        batch_size=batch_size,
        shuffle=False,
        gen_percent=0,
        contrastive_embedding=False,
    )
    
    trainloader_len = len(train_loader)

    if base_model_path:
        base_state_dict = load_file(base_model_path)
    else:
        base_state_dict = None
    
    model_config = {
        'base_model_config': base_model_config,
        'num_classes': len(batch_vocab),
    }
    
    # save model config to file
    if accelerator.is_main_process:
        os.makedirs(os.path.dirname(model_config_path), exist_ok=True)
        with open(model_config_path, "w") as f:
            json.dump(model_config, f, indent=4)

    model, optimizer, scheduler = create_training_state_finetune(
        model_config,
        init_lr,
        cosine_warmup_ratio_or_step,
        frozen_max_epochs * trainloader_len,
        trainable_base_model=False,  # Set to False to freeze base model initially
        base_state_dict=base_state_dict,
    )
    
    init_wandb(
        wandb_enabled,
        wandb_entity,
        wandb_project,
        wandb_config,
        accelerator,
        wandb_run_name=wandb_run_name,
        wandb_run_notes=args.notes,
    )
    
    load_test = False
    save_path = os.path.join(os.path.dirname(best_dir), "initial_model.pth")
    if not load_test:
        torch.save(model.state_dict(), save_path)
    else:
        initial_state_dict = torch.load(save_path)
        model.load_state_dict(initial_state_dict)

    train_loader, valid_loader, model, optimizer, scheduler = accelerator.prepare(
        train_loader, valid_loader, model, optimizer, scheduler
    )
    accelerator.save_model(model, best_dir)
    epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0

    while epoch < frozen_max_epochs:
        logger.info(f"Epoch {epoch + 1}/{frozen_max_epochs + unfrozen_max_epochs}")
        epoch_start_time = time.time()

        # Train the model
        val_loss, val_acc = finetune(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            epoch=epoch,
            log_interval=log_interval,
            vocab=vocab,
            accelerator=accelerator,
            optimizer=optimizer,
            scheduler=scheduler,
            best_dir=best_dir,
            best_val_loss=best_val_loss,
            loss_fn=loss_fn,
        )

        # Log metrics to wandb
        epoch_end_logs(epoch_start_time, epoch, val_loss=val_loss, val_accuracy=val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            logger.info(f"New best validation loss: {best_val_loss:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            logger.info("Early stopping triggered. Stopping training.")
            break

        epoch += 1
        accelerator.wait_for_everyone()
        
    # now, train with base model unfrozen
    accelerator.wait_for_everyone()
    new_state_dict = load_file(os.path.join(best_dir, "model.safetensors"))
    new_model, new_optimizer, new_scheduler = create_training_state_finetune(
        model_config,
        init_lr,
        cosine_warmup_ratio_or_step,
        unfrozen_max_epochs * trainloader_len,
        trainable_base_model=True,  # Set to False to freeze base model initially
        state_dict=new_state_dict,
    )
    # for i, (name, param) in enumerate(new_model.named_parameters()):
    #     logger.info(f"{i}: {name}, requires_grad={param.requires_grad}")
    new_model, new_optimizer, new_scheduler = accelerator.prepare(
        new_model, new_optimizer, new_scheduler
    )
    logger.info("Starting finetuning with base model unfrozen...")
    while epoch < frozen_max_epochs + unfrozen_max_epochs:
        logger.info(f"Epoch {epoch + 1}/{frozen_max_epochs + unfrozen_max_epochs}")
        epoch_start_time = time.time()

        # Train the model
        val_loss, val_acc = finetune(
            model=new_model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            epoch=epoch,
            log_interval=log_interval,
            vocab=vocab,
            accelerator=accelerator,
            optimizer=new_optimizer,
            scheduler=new_scheduler,
            best_dir=best_dir,
            best_val_loss=best_val_loss,
            loss_fn=loss_fn,
        )

        # Log metrics to wandb
        epoch_end_logs(epoch_start_time, epoch, val_loss=val_loss, val_accuracy=val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            logger.info(f"New best validation loss: {best_val_loss:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            logger.info("Early stopping triggered. Stopping training.")
            break

        epoch += 1

    logger.info("Training complete with best validation loss: {:.4f}".format(best_val_loss))    
    
    # evaluate on eval set... test differences
    probs, targets = get_class_probs(new_model, valid_loader, vocab.pad_index, accelerator)
    probs = probs.cpu().numpy()
    targets = targets.cpu().numpy()
    
    if accelerator.is_main_process:
        print("shape of eval probs and targets is:", probs.shape, targets.shape)
        print("location of eval probs and targets is", probs.device, targets.device)
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
        with open(os.path.join(args.best_dir, "valid_results.json"), "w") as f:
            json.dump(region_scores, f, indent=4)

        print(f"Scores for all regions saved to {args.best_dir}")
        
    # evaluate on train set
    accelerator.wait_for_everyone()
    probs, targets = get_class_probs(new_model, train_loader, vocab.pad_index, accelerator)
    probs = probs.cpu().numpy()
    targets = targets.cpu().numpy()
    
    if accelerator.is_main_process:
        print("shape of train probs and targets is:", probs.shape, targets.shape)
        print("location of train probs and targets is", probs.device, targets.device)
        predictions = np.argmax(probs, axis=1)
        total_accuracy = accuracy_score(targets, predictions)
        region_scores = []
        for region, index in batch_vocab.stoi.items():
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
        with open(os.path.join(args.best_dir, "train_results.json"), "w") as f:
            json.dump(region_scores, f, indent=4)

        print(f"Scores for all regions saved to {args.best_dir}")
        
    accelerator.end_training()
    
    
    
    
