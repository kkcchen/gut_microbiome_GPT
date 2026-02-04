"""
Model configuration and initialization utilities.
"""
import json
from omegaconf import OmegaConf
from trainers import logger
from model import hgmGPT

class DictStateWrapper:
    def __init__(self, data=None):
        self.data = data or {}

    def state_dict(self):
        return self.data

    def load_state_dict(self, state):
        self.data = state

def build_model_config(cfg, taxa_vocab, batch_vocab=None, graph_data=None) -> dict:
    """
    Construct complete model configuration from training config and data artifacts.
    
    :param cfg: OmegaConf configuration object.
    :param taxa_vocab: Microbiome vocabulary object.
    :param batch_vocab: Batch vocabulary (if using batch labels).
    :param graph_data: Graph data object (if using GNN).
    :return: Complete model configuration dictionary.
    """
    # Start with static model parameters from config
    model_config = OmegaConf.to_container(cfg.model.params, resolve=True)
    
    # Add dynamic configuration based on data and tasks
    tasks = OmegaConf.to_container(cfg.training.tasks, resolve=True)
    dynamic_config = {
        # use the vocab to initialize some parameters
        "num_taxa": len(taxa_vocab),
        "use_batch_labels": batch_vocab is not None,
        "num_batch_labels": len(batch_vocab) if batch_vocab else None,
        "tasks" : tasks,
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
        config=cfg,
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
    accelerator.register_for_checkpointing(model, optimizer, scheduler, extra_states)
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

    




# def restore_or_initialize_state(cfg, model_config, train_loader, accelerator):
#     """
#     Restore from checkpoint or initialize fresh training state.
    
#     :param cfg: Configuration object.
#     :param model_config: Model configuration dictionary.
#     :param train_loader: Training data loader (for computing total steps).
#     :param accelerator: Accelerator instance.
#     :return: Dictionary containing training state components.
#     """
#     from utils.config_utils import get_wandb_config
    
#     total_steps = len(train_loader) * cfg.training.max_epochs
    
#     model, optimizer, scheduler, epoch, best_val_loss, patience_counter, extra_state = \
#         create_or_restore_training_state_wandb(
#             model_config=model_config,
#             init_lr=cfg.training.init_lr,
#             warmup_ratio_or_step=cfg.training.cosine_warmup_ratio_or_step,
#             total_steps=total_steps,
#             checkpoint_dir=cfg.paths.checkpoint_dir,
#             use_wandb=cfg.wandb.enabled,
#             wandb_entity=cfg.wandb.get('entity', None),
#             wandb_project=cfg.wandb.get('project', 'microbiome-pretrain'),
#             wandb_config=get_wandb_config(cfg),
#             accelerator=accelerator,
#             wandb_run_name=cfg.wandb.get('run_name', None),
#             wandb_run_notes=cfg.wandb.get('run_notes', None)
#         )
    
#     return {
#         'model': model,
#         'optimizer': optimizer,
#         'scheduler': scheduler,
#         'epoch': epoch,
#         'best_val_loss': best_val_loss,
#         'patience_counter': patience_counter,
#         'extra_state': extra_state,
#         'train_loader': None,  # Will be filled by caller
#         'valid_loader': None,  # Will be filled by caller
#     }