import time
from typing import List, Dict, Any
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from models import TransformerModel, FinetunedTransformer
import os
import shutil
import json

from data_utils.preprocessor import Preprocessor
from data_utils.tokenizer import Tokenizer

from accelerate import Accelerator
from accelerate.utils import broadcast_object_list

import transformers
import anndata as ad

import wandb

from .custom_losses import (
    masked_relative_error,
    masked_mse_loss,
    env_contrastive_loss
)
import torch.nn.functional as F

from data_utils.vocab import MicrobiomeVocab, BatchVocab
from trainers import logger

# Define a simple state wrapper for the model to use with Accelerator
class DictStateWrapper:
    def __init__(self, data=None):
        self.data = data or {}

    def state_dict(self):
        return self.data

    def load_state_dict(self, state):
        self.data = state


def pretrain(
        model: TransformerModel, 
        train_loader: DataLoader,
        valid_loader: DataLoader,
        epoch: int,
        log_interval: int,
        vocab: MicrobiomeVocab,
        accelerator: Accelerator,
        optimizer,
        scheduler,
        use_batch_labels: bool,
        best_dir: str,
        use_mvc: bool,
        use_tcs: bool,
        use_contrastive: bool,
        # save_interval: int = -1,
        best_val_loss: float = float("inf"),
    ) -> None:
    """
    Train the model for one epoch.
    """
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    # total_cls = 0.0
    total_gen = 0.0
    # total_mvc = 0.0
    total_error = 0.0

    num_batches = len(train_loader)
    val_losses = []
    val_mres = []

    log_batch_start_time = time.time()

    for batch, data_dict in enumerate(train_loader):
        global_iter = epoch * num_batches + batch

        with accelerator.accumulate(model):
            if use_contrastive:
                data_dict_main = data_dict["view1"]
                data_dict_aux = data_dict["view2"]

                # View 2 (auxiliary view for contrastive learning)
                taxa_aux = data_dict_aux["ids"]
                taxa_target_aux = data_dict_aux["target_ids"]
                values_aux = data_dict_aux["corrupted_values"]
                key_padding_mask_aux = taxa_aux.eq(vocab.pad_index)
                values_target_aux = data_dict_aux["target_values"]
                known_positions_aux = values_target_aux.eq(vocab.mask_value)  # positions that are not masked
                positions_to_match_aux = (values_target != vocab.mask_value) & (values_target != vocab.pad_value)

            else:
                data_dict_main = data_dict

            # View 1 (main)
            taxa = data_dict_main["ids"]
            taxa_target = data_dict_main["target_ids"]
            values = data_dict_main["corrupted_values"]
            key_padding_mask = taxa.eq(vocab.pad_index)
            values_target = data_dict_main["target_values"]
            known_positions = values_target.eq(vocab.mask_value)  # positions that are not masked
            positions_to_match = (values_target != vocab.mask_value) & (values_target != vocab.pad_value)

            if use_batch_labels:
                batch_labels = data_dict["batch_labels"]
            else:
                batch_labels = None
            # else:
            #     input_gene_ids = data_dict["gene"]
            #     input_values = data_dict["masked_expr"]
            #     target_values = data_dict["expr"]
            #     src_key_padding_mask = input_gene_ids.eq(vocab[args.pad_token])

            with accelerator.autocast():
                # if USE_GENERATIVE_TRAINING:
                output_dict = model(
                    taxa,
                    values,
                    key_padding_mask,
                    known_positions=known_positions,
                    MVC=use_mvc,
                    TCS=use_tcs,
                    batch_labels=batch_labels,
                )
                abundance_preds = output_values = output_dict["preds"]

                loss_mse = masked_mse_loss(
                    abundance_preds, values_target, positions_to_match
                )
                loss = loss_mse
                accelerator.log({"train/loss_pcpt": loss_mse.item()}, step=global_iter)

                if use_mvc:
                    loss_mvc = masked_mse_loss(
                        output_dict["mvc_preds"],
                        values_target,
                        positions_to_match,
                    )
                    loss = loss + loss_mvc
                    accelerator.log({"train/mvc": loss_mvc.item()}, step=global_iter)
                # else: # if not using generative training
                #     output_dict = model(
                #         input_gene_ids,
                #         input_values,
                #         src_key_padding_mask=src_key_padding_mask,
                #         CLS=USE_CLS,
                #         CCE=USE_CCE,  # TODO: move these flags to model's attributes
                #         MVC=MVC,
                #         generative_training=False,
                #     )
                #     output_values = output_dict["mlm_output"]

                #     positions_to_match = input_values.eq(
                #         args.mask_value
                #     )  # the postions to predict
                #     loss = loss_mse = criterion(
                #         output_values, target_values, positions_to_match
                #     )
                #     writer.add_scalar("train/mse", loss_mse, global_iter)
                #     if USE_CLS:
                #         target_labels = data_dict["celltypes"]
                #         loss_cls = criterion_cls(output_dict["cls_output"], target_labels)
                #         loss = loss + loss_cls
                #         writer.add_scalar("train/cls", loss_cls, global_iter)
                if use_tcs:
                    flattened_preds = output_dict["taxa_preds"].view(-1, output_dict["taxa_preds"].shape[-1])
                    flattened_target = taxa_target.view(-1)
                    # loss_tcs = masked_mse_loss(
                    #     output_dict["taxa_preds"],
                    #     values_target,
                    #     positions_to_match,
                    # )
                    loss_tcs = F.cross_entropy(
                        flattened_preds,
                        flattened_target,
                        ignore_index=vocab.pad_index,
                    )
                    loss = loss + loss_tcs
                    accelerator.log({"train/tcs": loss_tcs.item()}, step=global_iter)
                if use_contrastive:
                    output_dict_aux = model(
                        taxa_aux,
                        values_aux,
                        key_padding_mask_aux,
                        known_positions=known_positions_aux,
                        # CLS=False,
                        MVC=False,
                        batch_labels=batch_labels,
                        # generative_training=True,
                    )
                    # gather doesn't work because it doesn't support backprop. do local contrastive loss only for now
                    # env1_all = accelerator.gather(output_dict["cell_emb"])
                    # env2_all = accelerator.gather(output_dict_aux["cell_emb"])
                    env1 = output_dict["cell_emb"]
                    env2 = output_dict_aux["cell_emb"]
                    loss_cce = 10 * env_contrastive_loss(
                        env1, env2, 0.5
                    )
                    loss = loss + loss_cce
                    accelerator.log({"train/cce": loss_cce.item()}, step=global_iter)
                    
                    abundance_preds_aux = output_dict_aux["preds"]
                    # output_values = (output_values + abundance_preds_aux) / 2

                    loss_mse_aux = masked_mse_loss(
                        abundance_preds_aux, values_target_aux, positions_to_match_aux
                    )
                    loss += loss_mse_aux
                    accelerator.log({"train/loss_pcpt_aux": loss_mse_aux.item()}, step=global_iter)
                #     if MVC:
                #         loss_mvc = criterion(
                #             output_dict["mvc_output"], target_values, positions_to_match
                #         )
                #         loss = loss + loss_mvc
                #         writer.add_scalar("train/mvc", loss_mvc, global_iter)

                # if USE_GENERATIVE_TRAINING and global_iter > 1000:
                # if global_iter > 500: # second pass with input_cell_embs
                #     previous_cell_embs = output_dict["cell_emb"].detach()
                #     output_dict = model(
                #         taxa,
                #         values,
                #         key_padding_mask,
                #         known_positions=known_positions,
                #         # CLS=False,
                #         MVC=False,
                #         batch_labels=batch_labels,
                #         input_cell_emb=previous_cell_embs,
                #         # generative_training=True,
                #     )
                #     preds = output_dict_aux["gen_preds"]
                #     loss_gen = masked_mse_loss(preds, gen_values_target, positions_to_match)
                #     loss = loss + loss_gen
                #     accelerator.log({"train/loss_gen": loss_gen.item()}, step=global_iter)

            # TODO: try this choice of using a separate backprop
            # # this part is for the choice of using a separate backprop
            # model.zero_grad()
            # scaler.scale(loss_gen).backward()
            # scaler.unscale_(optimizer)
            # torch.nn.utils.clip_grad_norm_(
            #     model.parameters(),
            #     1.0,
            #     error_if_nonfinite=False if scaler.is_enabled() else True,
            # )
            # scaler.step(optimizer)
            # scaler.update()

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        with torch.no_grad():
            mre = masked_relative_error(
                output_values, values_target, positions_to_match
            )
            accelerator.log({"train/mre": mre.item()}, step=global_iter)

        total_loss += loss.item()
        total_mse += loss_mse.item()
        # total_cls += loss_cls.item() if USE_CLS else 0.0
        total_gen += loss_gen.item() if "loss_gen" in locals() else 0.0
        # total_mvc += loss_mvc.item() if MVC else 0.0
        total_error += mre.item()
        # if args.local_rank in [0, -1] and batch % log_interval == 0 and batch > 0:
        if batch % log_interval == 0 and batch > 0:
            # Writer logs gradients distribution
            # for name, param in model.named_parameters():
            #     if param.requires_grad and param.grad is not None:
                    # writer.add_histogram(name + "_grad", param.grad, global_iter)
                    # writer.add_histogram(name + "_param", param, global_iter)

            # Log scalar values
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - log_batch_start_time) * 1000 / log_interval
            log_batch_start_time = time.time()
            cur_loss = total_loss / log_interval
            cur_mse = total_mse / log_interval
            # cur_cls = total_cls / log_interval if USE_CLS else 0.0
            cur_gen = total_gen / log_interval if "loss_gen" in locals() else 0.0
            # cur_mvc = total_mvc / log_interval if MVC else 0.0
            cur_error = total_error / log_interval
            # ppl = math.exp(cur_loss)
            logger.info(
                f"| epoch {epoch+1:3d} | {batch:3d}/{num_batches:3d} batches | "
                f"lr {lr:05.8f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {cur_loss:5.2f} | mse {cur_mse:5.2f} | mre {cur_error:5.2f} |"
                # + (f"cls {cur_cls:5.2f} | " if USE_CLS else "")
                + (f"gen {cur_gen:5.2f} |" if "loss_gen" in locals() else "")
                # + (f"mvc {cur_mvc:5.2f} |" if MVC else "")
            )

            accelerator.log({
                "learning_rate": scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else 0.0,
            }, step=global_iter)

            total_loss = 0
            total_mse = 0
            # total_cls = 0
            total_gen = 0
            # total_mvc = 0
            total_error = 0

        # immediately eval and save
        # if batch % save_interval == 0 and batch > 0:

        val_loss, val_mre = eval_and_save(
            model=model,
            valid_loader=valid_loader,
            best_dir=best_dir,
            vocab=vocab,
            best_val_loss=best_val_loss,
            global_iter=global_iter,
            accelerator=accelerator,
            use_batch_labels=use_batch_labels
            # save=(save_interval > 0 and batch % save_interval == 0),
        )

        best_val_loss = min(best_val_loss, val_loss)

        model.train()  # important, reset to train mode
        val_losses.append(val_loss)
        val_mres.append(val_mre)

    epoch_val_loss = np.mean(val_losses)
    epoch_val_mre = np.mean(val_mres)

    return epoch_val_loss, epoch_val_mre

def eval_and_save(
    model: nn.Module,
    valid_loader: DataLoader,
    best_dir: str,
    vocab: MicrobiomeVocab,
    best_val_loss: float,
    global_iter: int,
    accelerator: Accelerator,
    use_batch_labels: bool
    # save: bool = True,
) -> None:
    val_loss, val_mre = evaluate(model, valid_loader, vocab, accelerator, use_batch_labels).values()
    val_loss, val_mre = val_loss.item(), val_mre.item()

    logger.info(f"valid loss/mse {val_loss:5.4f} | mre {val_mre:5.4f}")
    accelerator.log({
        "val/val_loss": val_loss,
        "val/val_mre": val_mre,
    }, step=global_iter)

    if val_loss < best_val_loss:
        # save the best model
        logger.info(f"Saving the best model to {best_dir}")
        accelerator.save_model(model, best_dir)

    return val_loss, val_mre


def evaluate(
        model: nn.Module,
        valid_loader: DataLoader,
        vocab: MicrobiomeVocab,
        accelerator: Accelerator,
        use_batch_labels: bool
        ) -> Dict[str, torch.Tensor]:
    """
    Evaluate the model on the evaluation data.
    """
    model.eval()
    total_loss = 0.0
    total_error = 0.0
    with torch.no_grad():
        for data_dict in valid_loader:
            # if USE_GENERATIVE_TRAINING:
            taxa = data_dict["ids"]
            values = data_dict["corrupted_values"]
            key_padding_mask = taxa.eq(vocab.pad_index)
            values_target = data_dict["target_values"]
            known_positions = values_target.eq(vocab.mask_value)  # positions that are not masked
            positions_to_match = (values_target != vocab.mask_value) & (values_target != vocab.pad_value)
            if use_batch_labels:
                batch_labels = data_dict["batch_labels"]
            else:
                batch_labels = None
            # else:
            #     input_gene_ids = data_dict["gene"]
            #     input_values = data_dict["masked_expr"]
            #     target_values = data_dict["expr"]
            #     src_key_padding_mask = input_gene_ids.eq(vocab[args.pad_token])

            with accelerator.autocast():
                # if USE_GENERATIVE_TRAINING:
                output_dict = model(
                    taxa,
                    values,
                    key_padding_mask,
                    known_positions=known_positions,
                    batch_labels=batch_labels,
                    MVC=False,
                )
                abundance_preds = output_dict["preds"]
                # else:
                #     output_dict = model(
                #         input_gene_ids,
                #         input_values,
                #         src_key_padding_mask=src_key_padding_mask,
                #         CLS=False,  # evaluation does not need CLS or CCE
                #         CCE=False,
                #         MVC=False,
                #         generative_training=False,
                #     )
                #     output_values = output_dict["mlm_output"]
                #     positions_to_match = input_values.eq(args.mask_value)

            loss = masked_mse_loss(abundance_preds, values_target, positions_to_match)
            total_loss += loss.item()
            total_error += masked_relative_error(
                abundance_preds, values_target, positions_to_match
            ).item()

    total_loss = total_loss / len(valid_loader)
    total_error = total_error / len(valid_loader)
    return {
        "mse": torch.tensor(total_loss, dtype=torch.float),
        "mre": torch.tensor(total_error, dtype=torch.float),
    }


def epoch_end_logs(epoch_start_time, epoch, **metrics):
    elapsed = time.time() - epoch_start_time
    logger.info("-" * 89)
    
    metric_strs = [f"{key} {value:5.4f}" for key, value in metrics.items()]
    logger.info(
        f"| end of epoch {epoch + 1:3d} | time: {elapsed:5.2f}s | " + " | ".join(metric_strs)
    )

    logger.info(f"{'-' * 89}\n")

# we need to be careful when saving checkpoints since preemption can also
# occur during checkpointing. Therefore, we need to make sure the checkpoint
# file is either kept untouched or successfully updated during this process.
def commit_state(extra_state, epoch, best_val_loss, patience_counter, checkpoint_dir, accelerator: Accelerator):
    new_checkpoint_dir = os.path.join(checkpoint_dir, "new_checkpoint")
    actual_checkpoint_dir = os.path.join(checkpoint_dir, "actual_checkpoint")
    extra_state.data["epoch"] = epoch
    extra_state.data["best_val_loss"] = best_val_loss
    extra_state.data["patience_counter"] = patience_counter

    if os.path.exists(new_checkpoint_dir) and os.path.exists(actual_checkpoint_dir):
        shutil.rmtree(new_checkpoint_dir)
    os.makedirs(new_checkpoint_dir)
    accelerator.save_state(new_checkpoint_dir)

    # according to the GNU spec of rename, the state of checkpoint_dir
    # is atomic, i.e. it will either be modified or not modified, but not in
    # between, during a system crash (i.e. preemtion)
    if os.path.exists(actual_checkpoint_dir):
        shutil.rmtree(actual_checkpoint_dir)
    os.replace(new_checkpoint_dir, actual_checkpoint_dir)
    logger.info("Training state committed to {} at time {}".format(actual_checkpoint_dir, time.ctime(time.time())))


def create_or_restore_data_state(anndata_path, num_bins, restore_dir, accelerator: Accelerator, batch_obskey=None, nrows=None):
    print(f"Creating vocab and data from scratch using {anndata_path}")
    if accelerator.is_main_process:
        os.makedirs(restore_dir, exist_ok=True)
        
        batch_vocab = None
        
        if os.path.exists(os.path.join(restore_dir, "augmented_data.h5ad")):
            # load the vocab from the file
            adata = ad.read_h5ad(os.path.join(restore_dir, "augmented_data.h5ad"))
            assert "vocab_metadata" in adata.uns and "taxa_id" in adata.var
            vocab = MicrobiomeVocab.restore_vocab(adata)
            logger.info(f"Vocab restored from {restore_dir}")
            
            if batch_obskey:
                if f"{batch_obskey}_batch_vocab" in adata.uns:
                    batch_vocab = BatchVocab.restore_batchvocab(adata)
                    logger.info(f"Batch vocab restored from {restore_dir}")
                else:
                    batch_vocab = BatchVocab.create_batchvocab_from_scratch(batch_obskey, adata)
            
            # load the data state from the file
            assert "split" in adata.obs and "binned_rows" in adata.layers, "The AnnData object must have 'split' in obs."
            logger.info("Data state can be restored from {}".format(restore_dir))
        else:
            adata = ad.read_h5ad(anndata_path)
            if nrows:
                adata = adata[:nrows, :].copy()
            vocab = MicrobiomeVocab.create_vocab_from_scratch(adata)
            vocab.save_vocab(os.path.join(restore_dir, "vocab_file.json"))
            
            # make batch vocab
            if batch_obskey:
                batch_vocab = BatchVocab.create_batchvocab_from_scratch(batch_obskey, adata)
            
            # make data
            preprocessor = Preprocessor(
                binning=num_bins,
            )
            hmc_npy = np.array(adata.layers["top_512"].todense(), dtype=np.float32)
            taxa_ids = np.array(adata.var["taxa_id"])
            stacked_rows, _ = preprocessor.process_from_np(hmc_npy, taxa_ids)
            # create tokenizer
            adata.layers["binned_rows"] = stacked_rows
            
            # Randomly select exactly n_train indices without replacement
            train_indices = np.random.choice(adata.n_obs, size=int(adata.n_obs * 0.8), replace=False)
            is_train = np.zeros(adata.n_obs, dtype=bool)
            is_train[train_indices] = True            
            adata.obs["split"] = np.where(is_train, "train", "val")
        
        tokenizer = Tokenizer(vocab)
        data_dict = tokenizer.tokenize_and_pad_batch(adata, batch_obskey=batch_obskey)
        train_data_dict = {}
        valid_data_dict = {}
        # train and validation split
        split_keys = ["taxa_ids", "values"]

        if batch_obskey:
            split_keys.append("batch_labels")

        # Split data_dict based on adata.obs["split"]
        split_mask = adata.obs["split"].values
        train_mask = split_mask == "train"
        val_mask = split_mask == "val"
        for key in split_keys:
            train_data_dict[key] = data_dict[key][train_mask]
            valid_data_dict[key] = data_dict[key][val_mask]
        
        adata.write_h5ad(os.path.join(restore_dir, "augmented_data.h5ad"))
        print(f"length of train and valid data: {len(train_data_dict['taxa_ids'])}, {len(valid_data_dict['taxa_ids'])}")
        data_list = [train_data_dict, valid_data_dict, vocab, batch_vocab]
    else:
        data_list = [None, None, None, None]
    # broadcast to all other ranks
    accelerator.wait_for_everyone()
    broadcast_object_list(data_list)
    
    return tuple(data_list)


def create_or_restore_training_state_wandb(model_config, init_lr, warmup_ratio_or_step, total_steps, checkpoint_dir, wandb_enabled, wandb_entity, wandb_project, wandb_config, accelerator: Accelerator, wandb_run_name=None, wandb_run_notes=None):
    # initial configuration of the model
    # model = TransformerModel(
    #     d_model=512,
    #     nhead=8,
    #     d_hid=2048,
    #     nlayers=6,
    #     vocab=vocab,
    #     dropout=0.1,
    #     use_generative_training=True,
    # )

    model = TransformerModel(**model_config)
    trainable_params = model.parameters()
        
    optimizer = torch.optim.Adam(trainable_params, lr=init_lr)
    # setup scheduler
    # if warmup_ratio_or_step > 0:
    assert warmup_ratio_or_step > 0, "Warmup ratio or step must be positive"

    warmup_steps = (
        int(total_steps * warmup_ratio_or_step)
        if warmup_ratio_or_step < 1
        else int(warmup_ratio_or_step)
    )
    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # else:
    #     scheduler = torch.optim.lr_scheduler.StepLR(
    #         optimizer, scheduler_interval, gamma=args.scheduler_factor
    #     )
    # dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size,
    #                         sampler=StatefulSampler(dataset, shuffle=True),
    #                         num_workers=0)
    epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0

    extra_state = DictStateWrapper({
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "patience_counter": patience_counter,
    })

    # restore training state if checkpoint exists
    # need to be careful about temp/actual and preemption possibilities
    new_checkpoint_dir = os.path.join(checkpoint_dir, "new_checkpoint")
    actual_checkpoint_dir = os.path.join(checkpoint_dir, "actual_checkpoint")
    if not os.path.exists(new_checkpoint_dir) and not os.path.exists(actual_checkpoint_dir):
        logger.info("No checkpoint detected, starting from initial state")
        run = None
        if accelerator.is_main_process:
            run = wandb.init(
                mode="online" if wandb_enabled else "disabled",
                name=wandb_run_name,
                notes=wandb_run_notes,
                entity=wandb_entity,
                project=wandb_project,
                config=wandb_config,
                resume="allow"
            )
        accelerator.init_trackers(wandb_project)
        if run:
            extra_state.data["wandb_id"] = run.id if wandb_enabled else None
        accelerator.register_for_checkpointing(model, optimizer, scheduler, extra_state)
    else:
        if os.path.exists(new_checkpoint_dir):
            if os.path.exists(new_checkpoint_dir):
                shutil.rmtree(actual_checkpoint_dir)
            os.replace(new_checkpoint_dir, actual_checkpoint_dir)

        accelerator.register_for_checkpointing(model, optimizer, scheduler, extra_state)
        accelerator.load_state(actual_checkpoint_dir)
        epoch = extra_state.data.get('epoch', 0)
        best_val_loss = extra_state.data.get('best_val_loss', float("inf"))
        patience_counter = extra_state.data.get('patience_counter', 0)
        
        if accelerator.is_main_process:
            run_id = extra_state.data.get("wandb_id", None)
            run = wandb.init(
                id=run_id,
                resume="must" if run_id else None,
                mode="online" if wandb_enabled else "disabled",
                entity=wandb_entity,
                config=wandb_config,
                project=wandb_project,
            )
        accelerator.init_trackers(wandb_project)
        logger.info(f"Training state restored from actual_checkpoint at beginning of epoch {epoch + 1}")

    return model, optimizer, scheduler, epoch, best_val_loss, patience_counter, extra_state

