"""
Main trainer class for microbiome representation learning.
Handles training loop, validation, checkpointing, and logging.
Loss functions imported from separate module.
"""
import os
import time
import torch
from pathlib import Path
from typing import Dict, Optional
from trainers import logger
from trainers.loss_functions import *
from model import MASKING_TASKS, TAXA_MASKING_TASKS, ANY_MASKING_TASKS
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from omegaconf import OmegaConf
import numpy as np

class MicrobiomeTrainer:
    """
    Trainer for microbiome transformer pretraining.
    Manages training loop, validation, early stopping, and checkpointing.
    """
    
    def __init__(
        self,
        cfg,
        accelerator,
        taxa_vocab,
        batch_vocab: Optional[object] = None,
        graph_data: Optional[torch.Tensor] = None,
        class_weights: Optional[torch.Tensor] = None,
    ):
        """
        Initialize trainer with configuration and artifacts.

        :param cfg: Configuration object (OmegaConf).
        :param accelerator: HuggingFace Accelerator instance.
        :param taxa_vocab: Taxa vocabulary object.
        :param batch_vocab: Batch vocabulary object (optional).
        :param graph_data: Optional graph data for GNN (PyTorch Geometric Data).
        :param class_weights: Optional per-class loss weights for classification finetuning,
            e.g. to counter class imbalance (shape [num_classes]).
        """
        self.cfg = cfg
        self.accelerator = accelerator
        self.taxa_vocab = taxa_vocab
        self.batch_vocab = batch_vocab
        self.graph_data = graph_data
        self.class_weights = (
            class_weights.to(accelerator.device) if class_weights is not None else None
        )
        
        # Training config
        if 'grad_clip' not in cfg.training:
            logger.warning("TRAINING CONFIG: No grad_clip specified in config, defaulting to 1.0")
        if 'checkpoint_every' not in cfg.training:
            logger.warning("TRAINING CONFIG: No checkpoint_every specified in config, defaulting to 5 epochs")
        self.max_epochs = cfg.training.max_epochs
        self.patience = cfg.training.get('patience')
        if self.patience is None:
            self.patience = self.max_epochs
        self.log_interval = cfg.training.log_interval
        self.grad_clip = cfg.training.get('grad_clip', 1.0)
        self.checkpoint_every = cfg.training.get('checkpoint_every', 5)
        
        # Task flags
        self.use_batch_labels = cfg.data.use_batch_labels
        
        # Paths
        self.best_dir = cfg.paths.best_dir
        self.checkpoint_dir = cfg.paths.checkpoint_dir
        self.intermediate_dir = cfg.paths.intermediate_dir
        
        # wandb
        self.run_wandb = cfg.wandb.enabled
        self.global_step = 0
        
        
    
    def train(
        self,
        model,
        train_loader,
        valid_loader,
        optimizer,
        scheduler,
        start_epoch: int = 0,
        best_val_loss: float = float('inf'),
        patience_counter: int = 0
    ):
        """
        Main training loop.
        
        :param model: Model to train (already prepared by accelerator).
        :param train_loader: Training dataloader (already prepared).
        :param valid_loader: Validation dataloader (already prepared).
        :param optimizer: Optimizer (already prepared).
        :param scheduler: LR scheduler (already prepared).
        :param start_epoch: Epoch to start from (for resuming).
        :param best_val_loss: Best validation loss so far.
        :param patience_counter: Current patience counter for early stopping.
        """
        epoch = start_epoch
        
        logger.info("Starting training loop...")
        logger.info(f"Start epoch: {epoch}, Max epochs: {self.max_epochs}")
        logger.info(f"Best val loss: {best_val_loss:.6f}, Patience: {patience_counter}/{self.patience}")
        
        while epoch < self.max_epochs:
            logger.info("=" * 80)
            logger.info(f"Epoch {epoch + 1}/{self.max_epochs}")
            logger.info("=" * 80)
            epoch_start_time = time.time()
            
            # Train for one epoch
            train_metrics = self._train_epoch(
                model, train_loader, optimizer, scheduler, epoch
            )
            
            # Validate
            val_metrics = self._validate_epoch(model, valid_loader, epoch)
            
            # Log epoch summary
            self._log_epoch_summary(
                epoch, epoch_start_time, train_metrics, val_metrics
            )

            if self.run_wandb and self.accelerator.is_main_process:
                epoch_time = time.time() - epoch_start_time
                wandb_epoch_metrics = {
                    'epoch': epoch,
                    'epoch_time': epoch_time,
                    'train/epoch_loss': train_metrics['total_loss'],
                    'val/epoch_loss': val_metrics['total_loss'],
                    'learning_rate': scheduler.get_last_lr()[0],
                    'patience': patience_counter,
                    'best_val_loss': best_val_loss,
                }
                
                # Add task-specific metrics
                for key, value in train_metrics.items():
                    if key != 'total_loss':
                        wandb_epoch_metrics[f'train/epoch_{key}'] = value
                
                for key, value in val_metrics.items():
                    if key != 'total_loss':
                        wandb_epoch_metrics[f'val/epoch_{key}'] = value
                
                self.accelerator.log(wandb_epoch_metrics, step=self.global_step)
            
            # Get current validation loss
            val_loss = val_metrics['total_loss']
            
            # Early stopping and best model saving
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                logger.info(f"New best validation loss: {best_val_loss:.6f}")
                patience_counter = 0
                
                # Save best model - barrier before to ensure all processes are synced
                self.accelerator.wait_for_everyone()
                if self.accelerator.is_main_process:
                    self._save_best_model(model, epoch, best_val_loss)
            else:
                patience_counter += 1
                logger.info(f"No improvement. Patience: {patience_counter}/{self.patience}")
            
            # Check early stopping
            if patience_counter >= self.patience:
                logger.info("=" * 80)
                logger.info("Early stopping triggered. Stopping training.")
                logger.info("=" * 80)
                break
            
            # Save intermediate checkpoints at specified epochs
            if (epoch + 1) % self.checkpoint_every == 0:
                # Barrier before saving to ensure all processes have finished the epoch
                self.accelerator.wait_for_everyone()
                self._save_intermediate_checkpoint(model, epoch)
                if self.accelerator.is_main_process:
                    self._save_training_state(
                        model, optimizer, scheduler, epoch, best_val_loss, patience_counter
                    )
            

            # Update epoch
            epoch += 1
            # Standard barrier before next epoch
            self.accelerator.wait_for_everyone()

        logger.info("=" * 80)
        logger.info(f"Training complete! Best validation loss: {best_val_loss:.6f}")
        logger.info("=" * 80)
    
    def _train_epoch(
        self,
        model,
        train_loader,
        optimizer,
        scheduler,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        :param model: Model to train.
        :param train_loader: Training dataloader.
        :param optimizer: Optimizer.
        :param scheduler: LR scheduler.
        :param epoch: Current epoch number.
        :return: Dictionary of training metrics.
        """
        model.train()
        
        total_loss = 0.0
        total_metrics = {}
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Forward pass and compute losses
            loss, metrics = self._forward_step(model, batch)
            
            if not torch.isfinite(loss):
                print("LOSS IS NON-FINITE")
                raise RuntimeError("Non-finite loss")
            
            # Optimizer step - Item 8: zero_grad BEFORE backward
            optimizer.zero_grad()

            # Backward pass
            self.accelerator.backward(loss)
            
            # Gradient clipping - only if not using automatic gradient scaling from accelerator
            if self.grad_clip > 0:
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(model.parameters(), self.grad_clip)
            
            # Optimizer step
            optimizer.step()
            scheduler.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            for key, value in metrics.items():
                if key not in total_metrics:
                    total_metrics[key] = 0.0
                total_metrics[key] += value
            
            num_batches += 1
            self.global_step += 1
            if self.run_wandb and self.accelerator.is_main_process:
                wandb_step_metrics = {
                    'train/step_loss': loss.item(),
                    'train/learning_rate': scheduler.get_last_lr()[0],
                    'epoch': epoch + 1,
                }
                
                # Add task-specific metrics
                for key, value in metrics.items():
                    wandb_step_metrics[f'train/step_{key}'] = value
                
                self.accelerator.log(wandb_step_metrics, step=self.global_step)
            
            # Log at intervals
            if (batch_idx + 1) % self.log_interval == 0:
                # Gather metrics for logging
                loss_tensor = torch.tensor(loss.item(), device=self.accelerator.device)
                avg_loss_step = self.accelerator.gather_for_metrics(loss_tensor).mean().item()
                
                if self.accelerator.is_main_process:
                    avg_loss = total_loss / num_batches
                    current_lr = scheduler.get_last_lr()[0]
                    
                    log_str = (
                        f"Epoch {epoch + 1} | Batch {batch_idx + 1}/{len(train_loader)} | "
                        f"Step Loss: {avg_loss_step:.6f} | Avg Loss: {avg_loss:.6f} | LR: {current_lr:.2e}"
                    )
                    
                    # Add other metrics to log
                    for key, value in total_metrics.items():
                        if key != 'total_loss':
                            log_str += f" | {key}: {value / num_batches:.6f}"
                    
                    logger.info(log_str)
        
        # Compute epoch averages
        epoch_metrics = {
            'total_loss': total_loss / (num_batches if num_batches > 0 else 1),
        }
        
        for key, value in total_metrics.items():
            epoch_metrics[key] = value / (num_batches if num_batches > 0 else 1)
        
        return epoch_metrics
    
    def _validate_epoch(
        self,
        model,
        valid_loader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        :param model: Model to validate.
        :param valid_loader: Validation dataloader.
        :param epoch: Current epoch number.
        :return: Dictionary of validation metrics.
        """
        model.eval()
        
        total_loss = 0.0
        total_metrics = {}
        num_batches = 0
        
        with torch.no_grad():
            for batch in valid_loader:
                # Forward pass and compute losses
                loss, metrics = self._forward_step(model, batch)
                
                # Gather metrics from all processes
                loss_val = self.accelerator.gather_for_metrics(loss).mean().item()
                
                # Accumulate metrics
                total_loss += loss_val
                for key, value in metrics.items():
                    if key not in total_metrics:
                        total_metrics[key] = 0.0
                    # For metrics like accuracy, they should also be gathered
                    gathered_metric = self.accelerator.gather_for_metrics(torch.tensor(value, device=loss.device)).mean().item()
                    total_metrics[key] += gathered_metric
                
                num_batches += 1
        
        # Compute epoch averages
        epoch_metrics = {
            'total_loss': total_loss / num_batches,
        }
        
        for key, value in total_metrics.items():
            epoch_metrics[key] = value / num_batches
        
        return epoch_metrics
    
    def _forward_step(
        self,
        model,
        batch: Dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """
        Forward pass of model and loss computation in one function.
        
        Expected batch format:
        {
            'taxa_ids': (B, L) - selected taxa indices
            'normalized_counts': (B, L) - normalized counts (model input)
            'original_counts': (B, L) - original counts (reconstruction target)
            'expressed_mask': (B, L) - mask for expressed taxa
            'depth': (B,) - total count
            'original_depth': (B,) - original total count
            'batch_ids': (B,) - optional batch labels
            ... other metadata fields ...
        }

        :param model: Model to compute forward pass.
        :param batch: Batch dictionary from dataloader.
        :return: Tuple of (total_loss, metrics_dict).
        """
        # Prepare model inputs
        taxa_ids = batch['taxa_ids']  # (B, L)
        normalized_counts = batch['normalized_counts']  # (B, L)
        original_counts = batch['original_counts']  # (B, L)
        expressed_mask = batch['expressed_mask']  # (B, L)
        depth = batch['depth']  # (B,)
        original_depth = batch['original_depth']  # (B,)
        batch_ids = batch.get('batch_ids', None)  # (B,) or None
        # get metadata fields if specified
        metadata = {k: v for k, v in batch.items()
                    if k not in ['taxa_ids', 'normalized_counts', 'original_counts',
                                 'expressed_mask', 'depth', 'original_depth', 'batch_ids']}


        for k in ["taxa_ids", "normalized_counts", "depth"]:
            t = batch[k]
            if not torch.isfinite(t).all():
                raise ValueError(f"Non-finite in batch[{k}]")

        # ============= chunk for finetuning =============
        if self.cfg.training.get('finetune_mode', 'none') != 'none':
            # Use dedicated finetune_forward method
            predictions = model.finetune_forward(
                taxa_ids=taxa_ids,
                abundance_values=normalized_counts,
                depth=depth,
                batch_ids=batch_ids,
                graph_data=self.graph_data
            )
            
            # Get labels from batch
            labels = batch['labels']
            
            # Compute finetuning loss
            if model.finetune_task == 'classification':
                loss = F.cross_entropy(predictions, labels.long(), weight=self.class_weights)
                metrics = {'finetune_loss': loss.item()}
                
                # Add accuracy metric
                with torch.no_grad():
                    preds_class = predictions.argmax(dim=-1)
                    accuracy = (preds_class == labels).float().mean()
                    metrics['accuracy'] = accuracy.item()
            
            elif model.finetune_task == 'regression':
                loss = F.mse_loss(predictions, labels.float())
                metrics = {'finetune_loss': loss.item()}
                
                # Add MAE metric
                with torch.no_grad():
                    mae = torch.abs(predictions - labels).mean()
                    metrics['mae'] = mae.item()
            
            return loss, metrics
        # ============= END FINETUNING BRANCH =============
        
        # Forward pass through model
        # Adjust based on your model's forward signature
        outputs = {}
        if any(task not in ANY_MASKING_TASKS for task in self.cfg.training.tasks):
            outputs["normalized"] = model(
                taxa_ids=taxa_ids,
                abundance_values=normalized_counts,
                depth=depth,
                batch_ids=batch_ids,
                graph_data=self.graph_data,
            )
        if ANY_MASKING_TASKS.intersection(self.cfg.training.tasks):
            outputs_masked = model(
                taxa_ids=taxa_ids,
                abundance_values=normalized_counts,
                depth=depth,
                batch_ids=batch_ids,
                graph_data=self.graph_data,
            )
            outputs["original"] = outputs_masked
            
        
        
        # Compute loss using imported loss function
        loss, metrics = self.compute_loss(
            outputs=outputs,
            targets={
                'original_counts': original_counts,
                'normalized_counts': normalized_counts,
                'taxa_ids': taxa_ids,
                'expressed_mask': expressed_mask,
                'original_depth': original_depth,
                'batch_ids': batch_ids
            },
            cfg=self.cfg,
        )
        
        return loss, metrics
    
    def _log_epoch_summary(
        self,
        epoch: int,
        start_time: float,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ):
        """Log epoch summary statistics."""
        elapsed = time.time() - start_time
        
        logger.info("-" * 80)
        logger.info(f"Epoch {epoch + 1} Summary (Time: {elapsed:.2f}s)")
        logger.info("-" * 80)
        logger.info(f"Train Loss: {train_metrics['total_loss']:.6f} | Val Loss: {val_metrics['total_loss']:.6f}")
        
        # Log all other metrics
        metric_keys = set(train_metrics.keys()) | set(val_metrics.keys())
        metric_keys.discard('total_loss')
        
        for key in sorted(metric_keys):
            train_val = train_metrics.get(key, 0.0)
            val_val = val_metrics.get(key, 0.0)
            logger.info(f"Train {key}: {train_val:.6f} | Val {key}: {val_val:.6f}")
        
        logger.info("-" * 80)
    
    def _save_best_model(self, model, epoch: int, best_val_loss: float):
        """Save best model checkpoint."""
        save_path = Path(self.best_dir) / "best_model"
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model is a collective op in HF Accelerator, but here it's inside is_main_process guard
        # which is dangerous. HF save_model handles internal unwrap and main_process logic.
        # Moved to use accelerator.save_model outside the main process check if possible, or use torch.save on unwrapped.
        unwrapped_model = self.accelerator.unwrap_model(model)
        torch.save(unwrapped_model.state_dict(), save_path / "pytorch_model.bin")
        
        # Save metadata
        metadata = {
            'epoch': epoch,
            'best_val_loss': best_val_loss,
        }
        torch.save(metadata, save_path / 'metadata.pt')
        
        logger.info(f"Saved best model to {save_path}")
    
    def _save_intermediate_checkpoint(self, model, epoch: int):
        """Save intermediate checkpoint at specific epoch."""
        if self.accelerator.is_main_process:
            save_path = Path(self.intermediate_dir) / f"epoch_{epoch}"
            save_path.mkdir(parents=True, exist_ok=True)
            
            unwrapped_model = self.accelerator.unwrap_model(model)
            torch.save(unwrapped_model.state_dict(), save_path / "pytorch_model.bin")
            logger.info(f"Saved intermediate checkpoint to {save_path}")
    
    def _save_training_state(
        self,
        model,
        optimizer,
        scheduler,
        epoch: int,
        best_val_loss: float,
        patience_counter: int
    ):
        """Save full training state for resuming."""
        checkpoint_path = Path(self.checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        state = {
            'epoch': epoch,
            'best_val_loss': best_val_loss,
            'patience_counter': patience_counter,
            'model_state_dict': self.accelerator.unwrap_model(model).state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }
        
        torch.save(state, checkpoint_path / 'training_state.pt')
        logger.info(f"Saved training state to {checkpoint_path}")


    def compute_loss(self,   
                    outputs,
                    targets,
                    cfg,
                 ) -> tuple[torch.Tensor, Dict[str, float]]:
        '''
        Compute the loss for microbiome representation learning.
        :param outputs: Model outputs, dict containing the various different outputs
        :param targets: Ground truth targets, dict containing the various different targets
        :param cfg: Configuration object.
        :return: tuple of (loss tensor, metrics dictionary)
        '''
        tasks = self.cfg.training.tasks
        norm_strategy = self.cfg.data.norm_strategy
        loss = 0.0
        metrics = {}
        if any(task not in ANY_MASKING_TASKS for task in self.cfg.training.tasks):
            outputs_main = outputs["normalized"]
        if ANY_MASKING_TASKS.intersection(tasks):
            original = outputs["original"]

        # Expression reconstruction loss
        if 'denoising' in tasks:
            w = cfg.training.get('w_denoising', 1.0)
            denoising_loss = w*self._compute_denoising_loss(outputs_main, targets, cfg)

            metrics["denoising_loss"] = denoising_loss.item()
            loss += denoising_loss

        if MASKING_TASKS.intersection(tasks):
            masking_mask = original["masking_mask"].bool()
            is_binary = (norm_strategy == 'binary')
            is_binning = (norm_strategy == 'binning')
            log_transformed_targets = (norm_strategy == 'clr' or norm_strategy == 'log_rel_abundance' or norm_strategy == 'log_counts')

            if 'masking' in tasks:
                masking_logits = original["masking_logits"]
                if is_binary:
                    masking_loss = masked_binary_ce_loss(masking_logits, targets['original_counts'], masking_mask)

                    # Accuracy metric for binary
                    with torch.no_grad():
                        preds_binary = (masking_logits > 0).float()
                        targets_binary = (targets['original_counts'] > 0).float()
                        correct = ((preds_binary == targets_binary) * masking_mask).sum()
                        accuracy = correct / (masking_mask.sum() + 1e-6)
                        metrics['masking_binary_acc'] = accuracy.item()
                elif is_binning:
                    bin_target = targets['normalized_counts'] / cfg.data.get('num_bins', 15)
                    masking_loss = masked_mse_loss_counts(masking_logits, bin_target, masking_mask, log_transform=False)
                else:
                    masking_loss = masked_mse_loss(masking_logits, targets['original_counts'], masking_mask, log_transform=log_transformed_targets)
                loss += masking_loss
                metrics['masking_loss'] = masking_loss.item()

            if 'masking_from_cls' in tasks:
                # Same masked-reconstruction objective as 'masking', but predicted entirely
                # from the sample/cls token instead of per-position outputs, so the sample
                # embedding is pushed to encode information about the masked taxa.
                w = cfg.training.get('w_masking_from_cls', 1.0)
                cls_masking_logits = original["cls_masking_logits"]
                if is_binary:
                    masking_from_cls_loss = w * masked_binary_ce_loss(cls_masking_logits, targets['original_counts'], masking_mask)

                    with torch.no_grad():
                        preds_binary = (cls_masking_logits > 0).float()
                        targets_binary = (targets['original_counts'] > 0).float()
                        correct = ((preds_binary == targets_binary) * masking_mask).sum()
                        accuracy = correct / (masking_mask.sum() + 1e-6)
                        metrics['masking_from_cls_binary_acc'] = accuracy.item()
                elif is_binning:
                    bin_target = targets['normalized_counts'] / cfg.data.get('num_bins', 15)
                    masking_from_cls_loss = w * masked_mse_loss_counts(
                        cls_masking_logits, bin_target, masking_mask, log_transform=False
                    )
                else:
                    masking_from_cls_loss = w * masked_mse_loss(
                        cls_masking_logits, targets['original_counts'], masking_mask, log_transform=log_transformed_targets
                    )
                loss += masking_from_cls_loss
                metrics['masking_from_cls_loss'] = masking_from_cls_loss.item()

        if TAXA_MASKING_TASKS.intersection(tasks):
            w = cfg.training.get('w_masking_taxa', 1.0)
            masking_taxa_logits = original["masking_taxa_logits"]
            masking_taxa_mask = original["masking_taxa_mask"].bool()
            masking_taxa_loss = w * masked_ce_loss(masking_taxa_logits, targets['taxa_ids'], masking_taxa_mask)
            loss += masking_taxa_loss
            metrics['masking_taxa_loss'] = masking_taxa_loss.item()

            # Top-1 accuracy metric
            with torch.no_grad():
                preds_taxa = masking_taxa_logits.argmax(dim=-1)
                correct = ((preds_taxa == targets['taxa_ids']) * masking_taxa_mask).sum()
                accuracy = correct / (masking_taxa_mask.sum() + 1e-6)
                metrics['masking_taxa_acc'] = accuracy.item()
        return loss, metrics

    def _compute_denoising_loss(self, outputs, targets, cfg):
        '''
        Compute denoising loss.
        :param outputs: Model outputs for denoising.
        :param targets: Ground truth counts.
        :param cfg: Configuration object.
        :return: Denoising loss tensor.
        '''
        model_distr = cfg.model.params.model_distribution
        tasks = cfg.training.tasks
        # output from model will be different depending on modelling distribution
        if model_distr == 'dm':
            scale = outputs['dirichlet_scale']
            # for now just implement this from the sample embedding token
            # but should also be able to implement from the full transformer output
            mean_logits = outputs['denoising_mean']
            
            if not torch.isfinite(scale).all():
                raise RuntimeError("Non-finite scale before loss")
           
            denoising_loss = dm_nll_loss(scale,mean_logits, targets['original_counts'])
        
        else: # no distribution specified, direct count prediction
                # TODO: implement this both using the full transformer output and just the sample-embedding
                outputs_counts = outputs["denoising_pred"]
                # denoising_loss = mse_loss(
                denoising_loss = denoising_reconstruction_loss(
                                    outputs_counts,
                                    targets['original_counts']
                )
        
        # if 'denoising_from_token' in tasks:
        #     output_counts = outputs['denoising_projected']
        #     denoising_loss = denoising_reconstruction_loss(
        #         output_counts,
        #         targets['original_counts']
        #     )
        
        return denoising_loss

    def evaluate_on_test_set(
        self,
        model,
        test_loader,
        load_best_checkpoint: bool = True,
        best_model_path: Optional[Path] = None
    ) -> Dict[str, float]:
        """
        Evaluate finetuned model on test set with detailed metrics.
        
        :param model: Model to evaluate (prepared by accelerator).
        :param test_loader: Test dataloader (prepared by accelerator).
        :param load_best_checkpoint: Whether to load best checkpoint before evaluation.
        :param best_model_path: Path to best model checkpoint directory.
        :return: Dictionary of test metrics.
        """
        # load checkpoint if requested
        if load_best_checkpoint:
            if best_model_path is None:
                best_model_path = Path(self.best_dir) / "best_model"
            
            if not best_model_path.exists():
                logger.error(f"Best model not found at {best_model_path}")
                raise FileNotFoundError(f"Best model checkpoint not found at {best_model_path}")
            
            logger.info(f"Loading best model from {best_model_path}")
            unwrapped_model = self.accelerator.unwrap_model(model)
            
            # Load state dict
            state_dict_path = best_model_path / "pytorch_model.bin"
            if not state_dict_path.exists():
                state_dict_path = best_model_path / "model.pt"
            if not state_dict_path.exists():
                state_dict_path = best_model_path / "model.safetensors"

            if not state_dict_path.exists():
                raise FileNotFoundError(
                    f"No state dict found in {best_model_path}. "
                    f"Looked for pytorch_model.bin, model.pt, model.safetensors."
            )
    
            
            if state_dict_path.suffix == ".safetensors":
                from safetensors.torch import load_file as safetensors_load_file
                state_dict = safetensors_load_file(str(state_dict_path))  # returns a plain state_dict
            else:
                # PyTorch 2.6: weights_only defaults to True; add a safe fallback for trusted local files
                try:
                    state_dict = torch.load(state_dict_path, map_location="cpu")
                except Exception as e:
                    logger.warning(
                        f"torch.load(weights_only=True default) failed for {state_dict_path} "
                        f"({type(e).__name__}: {e}). Retrying with weights_only=False."
                    )
                    state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=False)
            unwrapped_model.load_state_dict(state_dict)
            model = self.accelerator.prepare(unwrapped_model)
            logger.info("Best model loaded successfully")
        
        logger.info("=" * 80)
        logger.info("EVALUATING ON TEST SET")
        logger.info("=" * 80)
        
        # Reuse _validate_epoch for basic evaluation
        test_metrics = self._validate_epoch(model, test_loader, epoch=-1)
        
        # Collect predictions and labels for detailed metrics
        model.eval()
        all_predictions = []
        all_probabilities = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                taxa_ids = batch['taxa_ids']
                normalized_counts = batch['normalized_counts']
                depth = batch['depth']
                batch_ids = batch.get('batch_ids', None)
                labels = batch['labels']

                # Forward pass
                predictions = model.finetune_forward(
                    taxa_ids=taxa_ids,
                    abundance_values=normalized_counts,
                    depth=depth,
                    batch_ids=batch_ids,
                    graph_data=self.graph_data
                )
                
                # Get predictions
                if model.finetune_task == 'classification':
                    preds = predictions.argmax(dim=-1)
                else:
                    preds = predictions.squeeze()
                
                all_predictions.append(self.accelerator.gather(preds).cpu().numpy())
                if model.finetune_task == 'classification':
                    probs = F.softmax(predictions, dim=-1)
                    all_probabilities.append(self.accelerator.gather(probs).cpu().numpy())
                all_labels.append(self.accelerator.gather(labels).cpu().numpy())
        
        all_predictions = np.concatenate(all_predictions)
        all_labels = np.concatenate(all_labels)
        if model.finetune_task == 'classification':
            all_probabilities = np.concatenate(all_probabilities)
        
        metrics = {
            'test_loss': test_metrics['total_loss'],
            'test_finetune_loss': test_metrics.get('finetune_loss', test_metrics['total_loss'])
        }
        
        if model.finetune_task == 'classification':
            metrics['test_accuracy'] = accuracy_score(all_labels, all_predictions)
            
            unique_labels = np.unique(all_labels)
            if len(unique_labels) == 2:
                metrics['test_f1'] = f1_score(all_labels, all_predictions, average='binary')
                metrics['test_auroc'] = roc_auc_score(all_labels, all_probabilities[:, 1])
            else:
                metrics['test_f1_macro'] = f1_score(all_labels, all_predictions, average='macro')
                metrics['test_f1_weighted'] = f1_score(all_labels, all_predictions, average='weighted')
                metrics['test_auroc_macro'] = roc_auc_score(
                    all_labels, 
                    all_probabilities, 
                    multi_class='ovr',
                    average='macro'
                )
                metrics['test_auroc_weighted'] = roc_auc_score(
                    all_labels, 
                    all_probabilities, 
                    multi_class='ovr',
                    average='weighted'
                )
                metrics['test_auroc_all'] = list(roc_auc_score(
                    all_labels, 
                    all_probabilities, 
                    multi_class='ovr',
                    average=None
                ))
            
            # Log results
            logger.info("=" * 80)
            logger.info("TEST SET RESULTS")
            logger.info("=" * 80)
            logger.info(f"Test Loss: {metrics['test_loss']:.4f}")
            logger.info(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
            if 'test_f1' in metrics:
                logger.info(f"Test F1: {metrics['test_f1']:.4f}")
            else:
                logger.info(f"Test F1 (macro): {metrics['test_f1_macro']:.4f}")
                logger.info(f"Test F1 (weighted): {metrics['test_f1_weighted']:.4f}")
                logger.info(f"Test AUROC (macro): {metrics['test_auroc_macro']:.4f}")
                logger.info(f"Test AUROC (weighted): {metrics['test_auroc_weighted']:.4f}")
                logger.info(f"Test AUROC (all classes): {metrics['test_auroc_all']}")
        elif model.finetune_task == 'regression':
            metrics['test_mae'] = mean_absolute_error(all_labels, all_predictions)
            metrics['test_mse'] = mean_squared_error(all_labels, all_predictions)
            metrics['test_rmse'] = np.sqrt(metrics['test_mse'])
            metrics['test_r2'] = r2_score(all_labels, all_predictions)
            
            # Log results
            logger.info("=" * 80)
            logger.info("TEST SET RESULTS")
            logger.info("=" * 80)
            logger.info(f"Test Loss: {metrics['test_loss']:.4f}")
            logger.info(f"Test MAE: {metrics['test_mae']:.4f}")
            logger.info(f"Test RMSE: {metrics['test_rmse']:.4f}")
            logger.info(f"Test R²: {metrics['test_r2']:.4f}")
        
        logger.info("=" * 80)
        
        # Save metrics
        if self.accelerator.is_main_process:
            
            def to_primitive(x):
                if isinstance(x, (np.floating, np.integer)):
                    return x.item()
                if isinstance(x, np.ndarray):
                    return x.tolist()
                if torch.is_tensor(x):
                    return x.detach().cpu().tolist() if x.ndim > 0 else x.item()
                if isinstance(x, dict):
                    return {k: to_primitive(v) for k, v in x.items()}
                if isinstance(x, (list, tuple)):
                    return [to_primitive(v) for v in x]
                return x

            metrics_primitive = to_primitive(metrics)
            
            
            test_metrics_path = Path(self.best_dir) / "test_metrics.yaml"
            OmegaConf.save(OmegaConf.create(metrics_primitive), test_metrics_path)
            logger.info(f"Saved test metrics to {test_metrics_path}")
        
        return metrics

