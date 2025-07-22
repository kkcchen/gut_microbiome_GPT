import time
from typing import List, Dict, Any
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from models import FinetunedTransformer
import transformers
import wandb

from accelerate import Accelerator

from data_utils.vocab import MicrobiomeVocab, BatchVocab
from trainers import logger

from safetensors.torch import load_file

def finetune(
        model: FinetunedTransformer, 
        train_loader: DataLoader,
        valid_loader: DataLoader,
        epoch: int,
        log_interval: int,
        vocab: MicrobiomeVocab,
        accelerator: Accelerator,
        optimizer,
        scheduler,
        best_dir: str,
        loss_fn,
        # save_interval: int = -1,
        best_val_loss: float = float("inf"),
        is_classification: bool = True,
    ):
    """
    Train the finetune model for one epoch.
    """
    model.train()
    total_loss = 0.0

    num_batches = len(train_loader)
    val_losses = []
    val_accs = []

    log_batch_start_time = time.time()

    for batch, data_dict in enumerate(train_loader):
        global_iter = epoch * num_batches + batch

        with accelerator.accumulate(model):                
            taxa = data_dict["ids"]
            values = data_dict["values"]
            key_padding_mask = taxa.eq(vocab.pad_index)
            targets = data_dict["batch_labels"] if is_classification else data_dict["continuous_labels"]

            with accelerator.autocast():
                output_dict = model(
                    taxa,
                    values,
                    src_key_padding_mask=key_padding_mask,
                )
                class_logits = output_dict["logits"]
                loss = loss_fn(class_logits, targets)
                accelerator.log({"train/loss_ce": loss.item()}, step=global_iter)

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            # Log scalar values
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - log_batch_start_time) * 1000 / log_interval
            log_batch_start_time = time.time()
            cur_loss = total_loss / log_interval
            logger.info(
                f"| epoch {epoch+1:3d} | {batch:3d}/{num_batches:3d} batches | "
                f"lr {lr:05.8f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {cur_loss:5.2f}"
            )

            accelerator.log({
                "learning_rate": scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else 0.0,
            }, step=global_iter)

            total_loss = 0

        # immediately eval and save
        # if batch % save_interval == 0 and batch > 0:

        val_loss, val_acc = eval_and_save(
            model=model,
            valid_loader=valid_loader,
            best_dir=best_dir,
            vocab=vocab,
            best_val_loss=best_val_loss,
            global_iter=global_iter,
            accelerator=accelerator,
            loss_fn=loss_fn,
        )

        best_val_loss = min(best_val_loss, val_loss)

        model.train()  # important, reset to train mode
        val_losses.append(val_loss)
        val_accs.append(val_acc)

    epoch_val_loss = np.mean(val_losses)
    epoch_val_acc = np.mean(val_accs)

    return epoch_val_loss, epoch_val_acc

def eval_and_save(
    model: FinetunedTransformer,
    valid_loader: DataLoader,
    best_dir: str,
    vocab: MicrobiomeVocab,
    best_val_loss: float,
    global_iter: int,
    accelerator: Accelerator,
    loss_fn,
) -> None:
    val_loss, val_acc = evaluate(model, valid_loader, vocab, accelerator, loss_fn).values()

    # logger.info(f"valid loss/mse {val_loss:5.4f} | accuracy {val_acc:5.4f}")
    accelerator.log({
        "val/val_loss": val_loss,
        "val/val_acc": val_acc,
    }, step=global_iter)

    if val_loss < best_val_loss:
        # save the best model
        logger.info(f"Saving the best model to {best_dir}")
        accelerator.save_model(model, best_dir)

    return val_loss, val_acc

def evaluate(
    model: FinetunedTransformer, 
    valid_loader: DataLoader,
    vocab: MicrobiomeVocab,
    accelerator: Accelerator,
    loss_fn,
    is_classification: bool = True,
    ) -> Dict[str, Any]:
    """
    Evaluate the model on the validation set.
    """
    model.eval()
    val_losses = []

    with torch.no_grad():
        correct_predictions = 0
        total_predictions = 0

        for data_dict in valid_loader:
            taxa = data_dict["ids"]
            values = data_dict["values"]
            key_padding_mask = taxa.eq(vocab.pad_index)
            targets = data_dict["batch_labels"] if is_classification else data_dict["continuous_labels"]

            with accelerator.autocast():
                output_dict = model(
                    taxa,
                    values,
                    src_key_padding_mask=key_padding_mask,
                )
                class_logits = output_dict["logits"]
                loss = loss_fn(class_logits, targets)
            val_losses.append(loss.item())

            # Calculate accuracy
            predictions = torch.argmax(class_logits, dim=-1)
            correct_predictions += (predictions == targets).sum().item()
            total_predictions += targets.size(0)

        avg_val_loss = np.mean(val_losses)
        accuracy = correct_predictions / total_predictions

    return {
        "val_loss": avg_val_loss,
        "val_acc": accuracy,
    }


def unfreeze_base_model(model: FinetunedTransformer, optimizer) -> List[nn.Parameter]:
    """
    Unfreeze the base model weights and return the list of trainable parameters.
    
    Args:
        model (FinetunedTransformer): The model whose base weights are to be unfrozen.
    
    Returns:
        List[nn.Parameter]: A list of trainable parameters in the base model.
    """
    model.set_base_model_trainable(trainable=True)
    optimizer.add_param_group({
        "params": model.base_model.parameters(),
    })
    

def load_model(
    model_config_path: str,
    model_path: str,
) -> FinetunedTransformer:
    """
    Load the finetuned transformer model from the best directory.
    
    Args:
        model_config_path (str): Path to the model configuration file.
        best_dir (str): Directory containing the best model state.
    
    Returns:
        FinetunedTransformer: The loaded model.
    """
    import json
    with open(model_config_path, "r") as f:
        model_config = json.load(f)
    
    model = FinetunedTransformer(model_config)
    state_dict = load_file(model_path)
    model.load_state_dict(state_dict)
    
    return model


def load_test_data(
    test_input: str,
    test_loc_labels_path: str,
) -> Dict[str, np.ndarray]:
    """
    Load the test data from the given input file and location labels.
    
    Args:
        test_input (str): Path to the test input .npy file (samples, taxa, 2).
        test_loc_labels_path (str): Path to the test location labels file.
    
    Returns:
        Dict[str, np.ndarray]: A dictionary containing the test data.
    """
    # Load the test input data
    test_data = np.load(test_input, allow_pickle=True)
    
    # Load the location labels
    loc_labels = np.load(test_loc_labels_path, allow_pickle=True)
    
    return {
        "test_data": test_data,
        "loc_labels": loc_labels,
    }
    
def create_training_state_finetune(model_config, init_lr, warmup_ratio_or_step, total_steps, trainable_base_model: bool, base_state_dict=None, state_dict=None):
    model = FinetunedTransformer(model_config)
    if base_state_dict:
        model.load_base_state_dict(base_state_dict)
    elif state_dict:
        model.load_state_dict(state_dict)
    else:
        logger.info("loading model from scratch!")
    trainable_params = model.set_base_model_trainable(trainable_base_model)
    optimizer = torch.optim.Adam(trainable_params, lr=init_lr)
    assert warmup_ratio_or_step > 0, "Warmup ratio or step must be positive"
    warmup_steps = (
        int(total_steps * warmup_ratio_or_step)
        if warmup_ratio_or_step < 1
        else int(warmup_ratio_or_step)
    )
    logger.info(f"len is {total_steps}")
    logger.info(f"warmup is {warmup_steps}")

    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    return model, optimizer, scheduler

def init_wandb(wandb_enabled, wandb_entity, wandb_project, wandb_config, accelerator: Accelerator, wandb_run_name=None, wandb_run_notes=None):
    if accelerator.is_main_process:
        wandb.init(
            mode="online" if wandb_enabled else "disabled",
            name=wandb_run_name,
            notes=wandb_run_notes,
            entity=wandb_entity,
            project=wandb_project,
            config=wandb_config,
            resume="allow"
        )
        accelerator.init_trackers(wandb_project)