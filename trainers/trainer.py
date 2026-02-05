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
from trainers.loss_functions import zinb_nll_loss, mse_loss, nt_xent_loss


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
        wandb_run = None,
    ):
        """
        Initialize trainer with configuration and artifacts.
        
        :param cfg: Configuration object (OmegaConf).
        :param accelerator: HuggingFace Accelerator instance.
        :param taxa_vocab: Taxa vocabulary object.
        :param batch_vocab: Batch vocabulary object (optional).
        :param graph_data: Optional graph data for GNN (PyTorch Geometric Data).
        """
        self.cfg = cfg
        self.accelerator = accelerator
        self.taxa_vocab = taxa_vocab
        self.batch_vocab = batch_vocab
        self.graph_data = graph_data
        
        # Training config
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
        self.wandb_run = wandb_run
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

            if self.wandb_run is not None and self.accelerator.is_main_process:
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
                
                self.wandb_run.log(wandb_epoch_metrics, step=self.global_step)
            
            # Get current validation loss
            val_loss = val_metrics['total_loss']
            
            # Early stopping and best model saving
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                logger.info(f"New best validation loss: {best_val_loss:.6f}")
                patience_counter = 0
                
                # Save best model
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
            if epoch % self.checkpoint_every == 0:
                self._save_intermediate_checkpoint(model, epoch)
                if self.accelerator.is_main_process:
                    self._save_training_state(
                        model, optimizer, scheduler, epoch, best_val_loss, patience_counter
                    )
            

            # Update epoch
            epoch += 1
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
            
            # Backward pass
            self.accelerator.backward(loss)
            
            # Gradient clipping
            if self.grad_clip > 0:
                self.accelerator.clip_grad_norm_(model.parameters(), self.grad_clip)
            
            # Optimizer step
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # Accumulate metrics
            total_loss += loss.item()
            for key, value in metrics.items():
                if key not in total_metrics:
                    total_metrics[key] = 0.0
                total_metrics[key] += value
            
            num_batches += 1
            self.global_step += 1
            if self.wandb_run is not None and self.accelerator.is_main_process:
                wandb_step_metrics = {
                    'train/step_loss': loss.item(),
                    'train/learning_rate': scheduler.get_last_lr()[0],
                    'epoch': epoch + 1,
                }
                
                # Add task-specific metrics
                for key, value in metrics.items():
                    wandb_step_metrics[f'train/step_{key}'] = value
                
                self.wandb_run.log(wandb_step_metrics, step=self.global_step)
            
            # Log at intervals
            if (batch_idx + 1) % self.log_interval == 0:
                avg_loss = total_loss / num_batches
                current_lr = scheduler.get_last_lr()[0]
                
                log_str = (
                    f"Epoch {epoch + 1} | Batch {batch_idx + 1}/{len(train_loader)} | "
                    f"Loss: {avg_loss:.6f} | LR: {current_lr:.2e}"
                )
                
                # Add other metrics to log
                for key, value in total_metrics.items():
                    if key != 'total_loss':
                        log_str += f" | {key}: {value / num_batches:.6f}"
                
                logger.info(log_str)
        
        # Compute epoch averages
        epoch_metrics = {
            'total_loss': total_loss / num_batches,
        }
        
        for key, value in total_metrics.items():
            epoch_metrics[key] = value / num_batches
        
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
                
                # Accumulate metrics
                total_loss += loss.item()
                for key, value in metrics.items():
                    if key not in total_metrics:
                        total_metrics[key] = 0.0
                    total_metrics[key] += value
                
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
            'perturbed_counts': (B, L) - perturbed counts (model input)
            'original_counts': (B, L) - original counts (reconstruction target)
            'expressed_mask': (B, L) - mask for expressed taxa
            'depth': (B,) - perturbed total count
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
        perturbed_counts = batch['perturbed_counts']  # (B, L)
        original_counts = batch['original_counts']  # (B, L)
        expressed_mask = batch['expressed_mask']  # (B, L)
        depth = batch['depth']  # (B,)
        original_depth = batch['original_depth']  # (B,)
        batch_ids = batch.get('batch_ids', None)  # (B,) or None
        # get metadata fields if specified
        metadata = {k: v for k, v in batch.items() 
                    if k not in ['taxa_ids', 'perturbed_counts', 'original_counts', 
                                 'expressed_mask', 'depth', 'original_depth', 'batch_ids']}
        
          
        
        # Forward pass through model
        # Adjust based on your model's forward signature
        outputs = model(
            taxa_ids=taxa_ids,
            abundance_values=perturbed_counts,
            depth=depth,
            batch_ids=batch_ids,
            graph_data=self.graph_data,
        )
        if 'contrastive' in self.cfg.training.tasks:
            perturbed_counts_2 = batch['perturbed_counts_2']  # (B, L)
            depth_2 = batch['depth_2']  # (B,)
            outputs_2 = model(
                taxa_ids=taxa_ids,
                abundance_values=perturbed_counts_2,
                depth=depth_2,
                batch_ids=batch_ids,
                graph_data=self.graph_data,
            )
            outputs = {"view_1": outputs, "view_2": outputs_2}
            
        
        
        # Compute loss using imported loss function
        loss, metrics = self.compute_loss(
            outputs=outputs,
            targets={
                'original_counts': original_counts,
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
        
        self.accelerator.save_model(model, save_path)
        
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
            
            self.accelerator.save_model(model, save_path)
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
        loss = 0.0
        metrics = {}
        if 'contrastive' in tasks:
            outputs_2 = outputs["view_2"]
            outputs = outputs["view_1"] # use this as outputs for any other losses
        
        # Expression reconstruction loss 
        if 'denoising' in tasks:
            # output from model will be different depending on modelling distribution
            if cfg.data.distribution == 'zinb':
                # ZINB distribution parameters
                outputs_mean = outputs["denoising_mean"]
                outputs_disp = outputs["denoising_disp"]
                outputs_pi = outputs["denoising_pi"]
                denoising_loss = zinb_nll_loss(
                    outputs_mean,
                    outputs_disp,
                    outputs_pi,
                    targets['original_counts'],
                )
            else: # no distribution specified, direct count prediction
                outputs_counts = outputs["denoising_pred"]
                denoising_loss = mse_loss(
                    outputs_counts,
                    targets['original_counts'],
                )
            metrics["denoising_loss"] = denoising_loss.item()
            loss += denoising_loss
        if 'contrastive' in tasks:
            # Placeholder for contrastive loss computation
            contrastive_loss = nt_xent_loss(
                outputs["projected"],
                outputs_2["projected"],   
                temperature=cfg.training.contrastive_temperature
            )
            metrics["contrastive_loss"] = contrastive_loss.item()
            loss += contrastive_loss
        return loss, metrics

    def _compute_denoising_loss(self, outputs, targets, cfg):
        '''
        Compute denoising loss.
        :param outputs: Model outputs for denoising.
        :param targets: Ground truth counts.
        :param cfg: Configuration object.
        :return: Denoising loss tensor.
        '''
        if cfg.data.distribution == 'zinb':
            outputs_mean = outputs["denoising_mean"]
            outputs_disp = outputs["denoising_disp"]
            outputs_pi = outputs["denoising_pi"]
            denoising_loss = zinb_nll_loss(
                outputs_mean,
                outputs_disp,
                outputs_pi,
                targets['original_counts'],
            )
        else: # no distribution specified, direct count prediction
            outputs_counts = outputs["denoising_counts"]
            denoising_loss = mse_loss(
                outputs_counts,
                targets['original_counts'],
            )
        return denoising_loss
    
    def inference(
        self,
        model,
        data_loader,
        embedding_type: str = "sample",
        return_outputs: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Run inference to extract embeddings for all samples.
        
        :param model: Trained model to run inference with.
        :param data_loader: DataLoader containing samples to embed.
        :param embedding_type: Type of embedding to extract:
            - "sample": Extract sample token embedding (first token)
            - "mean": Mean pool all taxa embeddings
            - "cls": Same as "sample" (alias)
            - "all": Return full sequence embeddings
        :param return_outputs: Whether to return full model outputs (for downstream tasks).
        :return: Dictionary containing:
            - 'embeddings': (N, d_model) tensor of sample embeddings
            - 'taxa_ids': (N, L) tensor of taxa IDs for each sample
            - 'batch_ids': (N,) tensor of batch IDs (if available)
            - 'sample_ids': List of sample identifiers
            - 'outputs': Full model outputs (if return_outputs=True)
        """
        model.eval()
        
        all_embeddings = []
        all_taxa_ids = []
        all_batch_ids = []
        all_sample_ids = []
        all_outputs = [] if return_outputs else None
        
        logger.info(f"Running inference on {len(data_loader)} batches...")
        logger.info(f"Embedding type: {embedding_type}")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                taxa_ids = batch['taxa_ids']  # (B, L)
                original_counts = batch['original_counts']  # (B, L)
                depth = batch['depth']  # (B,)
                batch_ids = batch.get('batch_ids', None)  # (B,) or None
                sample_ids = batch.get('sample_id', None)  # List of sample IDs
                
                # model inference
                sample_embeddings = model.inference(
                    taxa_ids=taxa_ids,
                    abundance_values=original_counts,
                    depth=depth,
                    batch_ids=batch_ids,
                    graph_data=self.graph_data
                )
                
                # Collect results
                all_embeddings.append(sample_embeddings.cpu())
                all_taxa_ids.append(taxa_ids.cpu())
                
                if batch_ids is not None:
                    all_batch_ids.append(batch_ids.cpu())
                
                if sample_ids is not None:
                    all_sample_ids.extend(sample_ids)
                
                if return_outputs:
                    all_outputs.append({k: v.cpu() for k, v in outputs.items()})
                
                # Log progress
                if (batch_idx + 1) % self.log_interval == 0:
                    logger.info(f"Processed {batch_idx + 1}/{len(data_loader)} batches")
        
        # Concatenate all batches
        embeddings = torch.cat(all_embeddings, dim=0)  # (N, d_model) or (N, num_tokens, d_model)
        taxa_ids = torch.cat(all_taxa_ids, dim=0)  # (N, L)
        
        results = {
            'embeddings': embeddings,
            'taxa_ids': taxa_ids,
        }
        
        if all_batch_ids:
            results['batch_ids'] = torch.cat(all_batch_ids, dim=0)
        
        if all_sample_ids:
            results['sample_ids'] = all_sample_ids
        
        if return_outputs:
            results['outputs'] = all_outputs
        
        logger.info(f"Inference complete! Extracted {embeddings.shape[0]} embeddings of dimension {embeddings.shape[-1]}")
        
        return results


    def save_embeddings(
        self,
        embeddings_dict: Dict[str, torch.Tensor],
        save_path: str,
        format: str = "pt"
    ):
        """
        Save extracted embeddings to disk.
        
        :param embeddings_dict: Dictionary returned from inference().
        :param save_path: Path to save embeddings.
        :param format: Save format - "pt" (PyTorch), "npz" (NumPy), or "h5" (HDF5).
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "pt":
            torch.save(embeddings_dict, save_path)
            logger.info(f"Saved embeddings to {save_path}")
        
        elif format == "npz":
            import numpy as np
            np_dict = {k: v.numpy() if isinstance(v, torch.Tensor) else v 
                    for k, v in embeddings_dict.items()}
            np.savez(save_path, **np_dict)
            logger.info(f"Saved embeddings to {save_path}")
        
        elif format == "h5":
            import h5py
            with h5py.File(save_path, 'w') as f:
                for key, value in embeddings_dict.items():
                    if isinstance(value, torch.Tensor):
                        f.create_dataset(key, data=value.numpy())
                    elif isinstance(value, list):
                        # Save string lists
                        dt = h5py.string_dtype(encoding='utf-8')
                        f.create_dataset(key, data=value, dtype=dt)
            logger.info(f"Saved embeddings to {save_path}")
        
        else:
            raise ValueError(f"Unknown format: {format}")

