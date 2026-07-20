"""
Model configuration and initialization utilities.
"""
import inspect
import json
import os
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from trainers import logger
from model import hgmGPT, MASKING_TASKS, TAXA_MASKING_TASKS
from typing import Dict, Optional

class DictStateWrapper:
    def __init__(self, data=None):
        self.data = data or {}

    def state_dict(self):
        return self.data

    def load_state_dict(self, state):
        self.data = state

def build_model_config(cfg, taxa_vocab, batch_vocab=None, graph_data=None, eval=False) -> dict:
    """
    Construct complete model configuration from training config and data artifacts.
    
    :param cfg: OmegaConf configuration object.
    :param taxa_vocab: Microbiome vocabulary object.
    :param batch_vocab: Batch vocabulary (if using batch labels).
    :param graph_data: Graph data object (if using GNN).
    :param eval: Whether this is for evaluation (affects certain config choices).
    :return: Complete model configuration dictionary.
    """
    # Start with static model parameters from config
    model_config = OmegaConf.to_container(cfg.model.params, resolve=True)
    
    # Add dynamic configuration based on data and tasks
    if not eval:
        tasks = OmegaConf.to_container(cfg.training.tasks, resolve=True)
    else:
        tasks = OmegaConf.to_container(cfg.eval.tasks, resolve=True)

    dynamic_config = {
        # use the vocab to initialize some parameters
        "num_taxa": len(taxa_vocab),
        "use_batch_labels": batch_vocab is not None,
        "num_batch_labels": len(batch_vocab) if batch_vocab else None,
        "tasks" : tasks,
        "num_gnn_nodes": graph_data.num_nodes if graph_data else None,
        "masking_prob": cfg.training.masking_prob if MASKING_TASKS.intersection(tasks) and not eval else None,
        "masking_taxa_prob": cfg.training.get('masking_taxa_prob', None) if TAXA_MASKING_TASKS.intersection(tasks) and not eval else None
    }
    
    model_config.update(dynamic_config)
    
    logger.info("Model configuration built:")
    logger.info(json.dumps(model_config, indent=2))
    
    return model_config

def initialize_optimizer(parameters, config):
    """
    Initialize optimizer based on configuration.
    
    :param parameters: Model parameters to optimize.
    :param config: Training configuration.
    :return: Initialized optimizer.
    """
    import torch.optim as optim
    
    optim_type = config.training.get('optimizer', 'adamw').lower()
    init_lr = config.training.init_lr
    
    if optim_type == 'adamw':
        optimizer = optim.AdamW(parameters, lr=init_lr)
    elif optim_type == 'sgd':
        optimizer = optim.SGD(parameters, lr=init_lr, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer type: {optim_type}")
    
    logger.info(f"Initialized {optim_type} optimizer with lr={init_lr}")
    return optimizer

def initialize_scheduler(optimizer, config, total_steps):
    """
    Initialize learning rate scheduler based on configuration.
    
    :param optimizer: Optimizer instance.
    :param config: Training configuration.
    :param total_steps: Total training steps for scheduler.
    :return: Initialized scheduler.
    """
    from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
    
    sched_type = config.training.get('lr_scheduler', 'cosine').lower()
    warmup_ratio_or_step = config.training.get('cosine_warmup_ratio_or_step', 0.1)
    
    if sched_type == 'cosine':
        if isinstance(warmup_ratio_or_step, float):
            warmup_steps = int(total_steps * warmup_ratio_or_step)
        else:
            warmup_steps = warmup_ratio_or_step
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
    elif sched_type == 'linear':
        if isinstance(warmup_ratio_or_step, float):
            warmup_steps = int(total_steps * warmup_ratio_or_step)
        else:
            warmup_steps = warmup_ratio_or_step
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {sched_type}")
    
    logger.info(f"Initialized {sched_type} scheduler with {warmup_steps} warmup steps.")
    return scheduler

def initialize_wandb(cfg, accelerator):
    """
    Initialize Weights & Biases logging if enabled.
    
    :param cfg: Configuration object.
    :param accelerator: Accelerator instance.
    :return: W&B run object or None.
    """
    if not cfg.wandb.enabled or not accelerator.is_main_process:
        return None
    
    import wandb
    wandb_project = cfg.wandb.get('project', 'microbiome-pretrain')
    wandb_run = wandb.init(
        project=wandb_project,
        entity=cfg.wandb.get('entity', None),
        name=cfg.wandb.get('run_name', None),
        notes=cfg.wandb.get('run_notes', None),
        config=OmegaConf.to_container(cfg, resolve=True),
        reinit=True
    )
    accelerator.init_trackers(wandb_project)
    
    logger.info(f"Initialized W&B run: {wandb_run.name}")
    return wandb_run


def initialize_training_components(model_config: dict, cfg, total_steps: int, accelerator):
    """
    Initialize model, optimizer, and scheduler.
    
    :param model_config: Complete model configuration.
    :param cfg: Training configuration.
    :param total_steps: Total training steps for scheduler.
    :param accelerator: Accelerator instance.
    :return: Tuple of (model, optimizer, scheduler).
    """
    # initialize model
    model = hgmGPT(**model_config)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"Model initialized with {sum(p.numel() for p in trainable_params)} trainable parameters.")
    # initialize optimizer based on config
    optimizer = initialize_optimizer(parameters=trainable_params,config=cfg)
    
    # load_state logic to handle resuming
    # checkpoint_path = cfg.paths.get('checkpoint_path', None)
    # if checkpoint_path and os.path.exists(checkpoint_path):
    #     logger.info(f"Loading training state from {checkpoint_path}...")
    #     raise NotImplementedError("State loading from checkpoint is not implemented yet. Please implement this logic.")

    # initialize scheduler based on config
    scheduler = initialize_scheduler(optimizer=optimizer, config=cfg, total_steps=total_steps)
    # initialize other training states
    epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0
    extra_states = DictStateWrapper({
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "patience_counter": patience_counter,
    })
    # initialize wandb
    wandb_run = initialize_wandb(cfg, accelerator=accelerator)
    extra_states.data['wandb_run'] = wandb_run if wandb_run else None
    accelerator.register_for_checkpointing(model)
    accelerator.register_for_checkpointing(optimizer)
    accelerator.register_for_checkpointing(scheduler)
    accelerator.register_for_checkpointing(extra_states)
    # collect and update status
    logger.info("Training components initialized.")
    return {
        'model': model,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'epoch': epoch,
        'best_val_loss': best_val_loss,
        'patience_counter': patience_counter,
        'extra_state': extra_states,
    }

def load_trained_model(cfg, model_config, accelerator):
    """
    Load trained model from checkpoint for inference.
    
    :param cfg: Configuration object.
    :param model_config: Complete model configuration.
    :param accelerator: Accelerator instance.
    :return: Loaded model.
    """
    # load config from cfg.paths.model_config_path if exists, otherwise build from current cfg
    model_config_path = cfg.paths.get('model_config_path', None)
    if model_config_path and os.path.exists(model_config_path):
        logger.info(f"Loading model configuration from {model_config_path}...")
        with open(model_config_path, 'r') as f:
            model_config = json.load(f)
        # Older checkpoints' saved model_config.json can contain keys that
        # have since been removed from hgmGPT's signature -- drop them
        # instead of failing, so any past checkpoint remains loadable.
        valid_keys = set(inspect.signature(hgmGPT.__init__).parameters) - {'self'}
        unknown_keys = set(model_config) - valid_keys
        if unknown_keys:
            logger.warning(
                f"Ignoring keys in {model_config_path} no longer accepted by hgmGPT: {sorted(unknown_keys)}"
            )
            model_config = {k: v for k, v in model_config.items() if k in valid_keys}
    else:
        logger.warning(f"Model configuration file not found at {model_config_path}. Building model configuration from current cfg.")

    model = hgmGPT(**model_config)
    checkpoint_path = cfg.paths.checkpoint_path
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    logger.info(f"Loading model checkpoint from {checkpoint_path}...")
    if checkpoint_path.endswith('.pt') or checkpoint_path.endswith('.pth') or checkpoint_path.endswith('.bin'):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    elif checkpoint_path.endswith('.safetensors'):
        from safetensors.torch import load_file
        checkpoint = load_file(checkpoint_path)
    else:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")

    if 'model_state_dict' in checkpoint:
        loaded_state = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        loaded_state = checkpoint['model']
    elif 'state_dict' in checkpoint:
        loaded_state = checkpoint['state_dict']
    else:
        loaded_state = checkpoint

    model.load_state_dict(loaded_state)
    model.to(accelerator.device)
    model.eval()
    
    logger.info("Model loaded and set to evaluation mode.")
    return model


def inference(
        model,
        data_loader,
        graph_data=None,
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
                normalized_counts = batch['normalized_counts']  # (B, L)
                depth = batch['depth']  # (B,)
                batch_ids = batch.get('batch_ids', None)  # (B,) or None
                sample_ids = batch.get('sample_id', None)  # List of sample IDs

                # model inference
                sample_embeddings = model.inference(
                    taxa_ids=taxa_ids,
                    abundance_values=normalized_counts,
                    depth=depth,
                    batch_ids=batch_ids,
                    graph_data=graph_data
                )
                
                # Collect results
                all_embeddings.append(sample_embeddings.cpu())
                all_taxa_ids.append(taxa_ids.cpu())
                
                if batch_ids is not None:
                    all_batch_ids.append(batch_ids.cpu())
                
                if sample_ids is not None:
                    all_sample_ids.extend(sample_ids)
                
        
        # Concatenate all batches
        embeddings = torch.cat(all_embeddings, dim=0)  # (N, d_model) or (N, num_tokens, d_model)
        # Batches are padded independently by the collator (to each batch's own
        # max expressed-taxa count), so taxa_ids sequence length L can differ
        # across batches -- pad to the global max before concatenating.
        max_seq_len = max(t.shape[1] for t in all_taxa_ids)
        all_taxa_ids = [
            F.pad(t, (0, max_seq_len - t.shape[1]), value=0) for t in all_taxa_ids
        ]
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
    

def load_pretrained_model_for_finetune(cfg, model_config, accelerator):
    """
    Load pretrained model weights and configure for finetuning.
    
    :param cfg: Configuration object.
    :param model_config: Model configuration dictionary.
    :param accelerator: Accelerator instance.
    :return: Model with loaded pretrained weights.
    """
    logger.info("=" * 80)
    logger.info("LOADING PRETRAINED MODEL")
    logger.info("=" * 80)
    
    # Initialize model with finetuning configuration
    model = hgmGPT(**model_config)
    
    # Load pretrained weights
    checkpoint_path = cfg.paths.get('checkpoint_path', None)
    if checkpoint_path:
        logger.info(f"Loading pretrained weights from: {checkpoint_path}")
        
        # Handle different checkpoint formats
        # checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if checkpoint_path.endswith('.pt') or checkpoint_path.endswith('.pth') or checkpoint_path.endswith('.bin'):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        elif checkpoint_path.endswith('.safetensors'):
            from safetensors.torch import load_file
            checkpoint = load_file(checkpoint_path)

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Load weights (strict=False to allow new finetuning head)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            logger.info(f"Missing keys (expected for new finetuning head): {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys: {unexpected_keys}")
        
        logger.info("✓ Loaded pretrained weights successfully")
    else:
        logger.info("=" * 80)
        logger.info("RANDOM WEIGHT INITIALIZATION - NO PRETRAINED CHECKPOINT")
        logger.info("Model will learn the downstream task entirely from scratch.")
        logger.info("=" * 80)

    # Configure parameter freezing based on finetune_mode
    finetune_mode = cfg.training.finetune_mode
    if not checkpoint_path and finetune_mode == "none":
        raise ValueError(
            "finetune_mode='none' freezes all parameters, but no pretrained checkpoint "
            "was provided, so the model is randomly initialized. Training would learn "
            "nothing. Use finetune_mode='full' or 'partial' when training from scratch."
        )
    logger.info(f"Finetuning mode: {finetune_mode}")
    
    # Log trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
               f"({100 * trainable_params / total_params:.2f}%)")
    
    return model

def initialize_finetuning_components(model, cfg, total_steps):
    """
    Initialize optimizer and scheduler for finetuning.
    
    :param model: Model to optimize.
    :param cfg: Configuration object.
    :param total_steps: Total training steps.
    :return: Dictionary with optimizer and scheduler.
    """
    # Create optimizer (only for trainable parameters)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    
    # optimizer = torch.optim.AdamW(
    #     trainable_params,
    #     lr=cfg.training.init_lr,
    #     weight_decay=cfg.training.weight_decay,
    #     betas=(cfg.training.get('adam_beta1', 0.9), 
    #            cfg.training.get('adam_beta2', 0.999))
    # )
    
    optimizer = initialize_optimizer(parameters=trainable_params,config=cfg)

    # Create scheduler
    from transformers import get_linear_schedule_with_warmup

    warmup_ratio_or_step = cfg.training.cosine_warmup_ratio_or_step
    if isinstance(warmup_ratio_or_step, float):
        warmup_steps = int(total_steps * warmup_ratio_or_step)
    else:
        warmup_steps = warmup_ratio_or_step

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    logger.info(f"Optimizer: AdamW (lr={cfg.training.init_lr})")
    logger.info(f"Scheduler: Linear warmup ({warmup_steps} steps) + decay")
    
    return {
        'optimizer': optimizer,
        'scheduler': scheduler
    }