from torch.utils.data import DataLoader
import torch

import os
import shutil
import numpy as np
import time

from data_utils.preprocessor import Preprocessor
from data_utils.dataloader import prepare_dataloader

from trainers.train_functions import (
    pretrain, commit_state, create_or_restore_data_state, create_or_restore_training_state_wandb, epoch_end_logs
)
from trainers import logger

from accelerate import Accelerator

import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train TransformerModel on microbiome data")
    parser.add_argument("--model-config-path", type=str, required=True, help="Path to model configuration file")
    parser.add_argument("--ann-table-path", type=str, required=True, help="Path to HMC table file")
    parser.add_argument("--vocab-path", type=str, default=None, help="Path to vocabulary file (if any), e.g. vocab created from evo2")
    parser.add_argument("--best-dir", type=str, required=True, help="Directory to save best model so far")
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="Directory to save checkpoints for preemption")
    parser.add_argument("--data-restore-dir", type=str, required=True, help="Directory to restore data state")
    parser.add_argument("--intermediate-dir", type=str, required=True, help="Directory to store intermediate checkpoints")

    # wandb
    parser.add_argument("--wandb-enabled", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-entity", type=str, default=None, help="wandb entity name")
    parser.add_argument("--wandb-project", type=str, default=None, help="wandb project name")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="wandb run name")
    parser.add_argument("--wandb-run-notes", type=str, default=None, help="wandb run notes")

    # optional training arguments
    parser.add_argument("--init-lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--max-epochs", type=int, default=15, help="Maximum number of epochs")
    parser.add_argument("--cosine-warmup-ratio-or-step", type=float, default=0.1, help="Scheduler warmup ratio or step")
    parser.add_argument("--log-interval", type=int, default=10, help="Interval for logging")
    parser.add_argument("--patience", type=int, default=None, help="Patience for early stopping")
    parser.add_argument("--grad-accumulation-steps", type=int, default=1, help="Number of gradient accumulation steps")
    parser.add_argument("--enable-fp16", action="store_true", help="Enable mixed precision training (FP16)")
    
    # model parameters
    parser.add_argument("--num-bins", type=int, default=15, help="Number of bins for binning")
    parser.add_argument("--use-batch-labels", action="store_true", help="Use batch labels in the dataset")
    parser.add_argument("--do-mvc", action="store_true", help="do mvc task for pretraining")
    parser.add_argument("--do-taxa-decoder", action="store_true", help="do taxa task for pretraining")
    parser.add_argument("--do-contrastive", action="store_true", help="Use contrastive embedding in the model")
    parser.add_argument("--train-mask-ratio", type=float, default=0.15, help="train mask ratio")
    parser.add_argument("--freeze-vocab", action="store_true", help="Freeze the embedding layer of the vocab, if initialized from a pre-trained embedding")
    parser.add_argument("--freeze-value-encoder", action="store_true", help="Freeze the value encoder layer")
    parser.add_argument("--data-bin-strategy", type=str, default="binning", choices=["binning", "clr", "clr_plus"], help="Data preprocessing strategy: 'binning' or 'clr'")
    parser.add_argument("--use-gnn", action="store_true", help="Use GNN embeddings as input features")
    parser.add_argument("--gnn-type", default=None, choices=[None, "gat", "gcn"], help="gnn type")


    # for debugging
    parser.add_argument("--nrows", type=int, default=None, help="For debugging to limit number of samples in set")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--start-over", action="store_true", help="Start over from scratch, ignoring existing data and checkpoints")
    parser.add_argument("--notes", type=str, default="", help="Notes for the current training run")
    
    args = parser.parse_args()

    model_config_path = args.model_config_path
    ann_table_path = args.ann_table_path
    vocab_path = args.vocab_path
    best_dir = args.best_dir
    checkpoint_dir = args.checkpoint_dir
    data_restore_dir = args.data_restore_dir

    wandb_enabled = args.wandb_enabled
    wandb_entity = args.wandb_entity
    wandb_project = args.wandb_project
    wandb_run_name = args.wandb_run_name
    wandb_run_notes = args.wandb_run_notes

    init_lr = args.init_lr
    batch_size = args.batch_size
    max_epochs = args.max_epochs
    cosine_warmup_ratio_or_step = args.cosine_warmup_ratio_or_step
    num_bins = args.num_bins
    log_interval = args.log_interval
    patience = args.patience if args.patience else max_epochs
    grad_accumulation_steps = args.grad_accumulation_steps
    enable_fp16 = args.enable_fp16
    do_mvc = args.do_mvc
    do_taxa_decoder = args.do_taxa_decoder
    do_contrastive = args.do_contrastive
    train_mask_ratio = args.train_mask_ratio
    freeze_vocab = args.freeze_vocab
    freeze_value_encoder = args.freeze_value_encoder
    bin_strategy = args.data_bin_strategy
    
    use_batch_labels = args.use_batch_labels
    nrows = args.nrows
    
    if args.use_gnn:
        assert args.gnn_type is not None, "Please specify --gnn-type when --use-gnn is set"
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    wandb_config={
        "do_mvc": do_mvc,
        "do_taxa_decoder": do_taxa_decoder,
        "do_contrastive": do_contrastive,
        "num_bins": num_bins,
        "use_batch_labels": use_batch_labels,
    }

    accelerator = Accelerator(gradient_accumulation_steps=grad_accumulation_steps, mixed_precision="fp16" if enable_fp16 else "no", log_with="wandb" if wandb_enabled else None)

    # Save notes to a markdown file in the save directory
    if accelerator.is_main_process and args.notes:
        notes_path = os.path.join(best_dir, "training_notes.md")
        os.makedirs(best_dir, exist_ok=True)
        with open(notes_path, "w") as notes_file:
            notes_file.write(args.notes)

    if args.start_over:
        logger.info("Starting over from scratch, deleting existing training state.")
        if os.path.exists(data_restore_dir):
            shutil.rmtree(data_restore_dir)
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)

    # Create or restore data state
    train_data_dict, valid_data_dict, vocab, batch_vocab, graph_data = create_or_restore_data_state(
        ann_table_path, 
        num_bins, 
        data_restore_dir, 
        accelerator, 
        batch_obskey="study_id" if use_batch_labels else None, 
        nrows=nrows,
        bin_strategy=bin_strategy,
        use_gnn=args.use_gnn
    )

    logger.info("Preparing dataloaders...")
    train_loader = prepare_dataloader(
        train_data_dict,
        use_batch_labels=use_batch_labels,
        use_continuous_labels=False,
        mask_ids=do_taxa_decoder,
        vocab=vocab,
        batch_size=batch_size,
        gen_percent=train_mask_ratio,
        shuffle=True,
        contrastive_embedding=do_contrastive,
    )
    valid_loader = prepare_dataloader(
        valid_data_dict,
        use_batch_labels=use_batch_labels,
        use_continuous_labels=False,
        vocab=vocab,
        batch_size=batch_size,
        gen_percent=0.15,
        shuffle=False,
        contrastive_embedding=False,
    )

    # Create or restore training state
    model_config = {
        "d_model": 128,
        "nhead": 8,
        "d_hid": 512,
        "nlayers": 3,
        "use_batch_labels": use_batch_labels,
        "dropout": 0.1,
        "n_input_bins": num_bins,
        "do_mvc": do_mvc,
        "do_taxa_decoder": do_taxa_decoder,
        "do_attn_mask": False,
        "vocab_len": len(vocab),
        "vocab_num_special_tokens": vocab.num_special_tokens,
        "vocab_pad_index": vocab.pad_index,
        "vocab_pad_value": vocab.pad_value,
        "vocab_mask_value": vocab.mask_value,
        "num_batch_labels": len(batch_vocab) if use_batch_labels else 0,
        "init_vocab_path": vocab_path,
        "freeze_vocab": freeze_vocab,
        "freeze_value_encoder": freeze_value_encoder,
        "input_emb_style": "scaling",
        # "input_emb_style": "continuous",
        "use_gnn": args.use_gnn,
        "num_gnn_nodes": graph_data.num_nodes if args.use_gnn else None,
        "gnn_type": args.gnn_type,
        "gnn_num_layers": 3 if args.use_gnn else None,
        "bin_strategy": bin_strategy,
    }
    logger.info(model_config)
    import json
    if accelerator.is_main_process:
        os.makedirs(os.path.dirname(model_config_path), exist_ok=True)
        with open(model_config_path, "w") as f:
            json.dump(model_config, f, indent=4)

    total_steps = len(train_loader) * max_epochs
    model, optimizer, scheduler, epoch, best_val_loss, patience_counter, extra_state = create_or_restore_training_state_wandb(
        model_config,
        init_lr,
        cosine_warmup_ratio_or_step,
        total_steps,
        checkpoint_dir,
        wandb_enabled,
        wandb_entity,
        wandb_project,
        wandb_config,
        accelerator=accelerator,
        wandb_run_name=wandb_run_name,
        wandb_run_notes=wandb_run_notes
    )

    train_loader, valid_loader, model, optimizer, scheduler = accelerator.prepare(
        train_loader, valid_loader, model, optimizer, scheduler
    )
    
    checkpoint_at = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14}

    while epoch < max_epochs:
        if epoch in checkpoint_at: # Save model every few epochs
            subdir = os.path.join(args.intermediate_dir, f"epoch_{epoch}")
            logger.info(f"Saving an intermediate checkpoint to {subdir}")
            accelerator.save_model(model, subdir)
            
        logger.info(f"Epoch {epoch + 1}/{max_epochs}")
        epoch_start_time = time.time()

        # Train the model
        val_loss, val_mre = pretrain(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            epoch=epoch,
            log_interval=log_interval,
            vocab=vocab,
            accelerator=accelerator,
            optimizer=optimizer,
            scheduler=scheduler,
            use_batch_labels=use_batch_labels,
            best_dir=best_dir,
            best_val_loss=best_val_loss,
            use_mvc=do_mvc,
            use_tcs=do_taxa_decoder,
            use_contrastive=do_contrastive,
            graph_data=graph_data if args.use_gnn else None,
        )

        # Log metrics to wandb
        epoch_end_logs(epoch_start_time, epoch, val_loss=val_loss, val_mre=val_mre)

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
        if accelerator.is_main_process:
            commit_state(extra_state, epoch, best_val_loss, patience_counter, checkpoint_dir, accelerator)
            
        accelerator.wait_for_everyone()

    logger.info("Training complete with best validation loss: {:.4f}".format(best_val_loss))
    accelerator.end_training()