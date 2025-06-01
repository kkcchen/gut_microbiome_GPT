import time
from typing import List, Dict, Any
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from models import TransformerModel
import os
import json

from data_utils.preprocessor import Preprocessor
from data_utils.dataloader import prepare_dataloader
from data_utils.tokenizer import Tokenizer

from sklearn.model_selection import train_test_split
import transformers

import wandb

from .custom_losses import (
    masked_relative_error,
    masked_mse_loss,
)

from data_utils import MicrobiomeVocab
from trainers import logger

# to do: convert tensorboard writer to wandb
def pretrain(
        model: nn.Module, 
        train_loader: DataLoader,
        valid_loader: DataLoader,
        epoch: int,
        log_interval: int,
        vocab: MicrobiomeVocab,
        enable_fp16: bool,
        grad_accu_steps: int,
        scaler,
        optimizer,
        scheduler,
        save_dir: str,
        device: str,
        logger,
        epoch_start_time: float,
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

    for batch, data_dict in enumerate(train_loader):
        global_iter = epoch * num_batches + batch

        data_dict = {k: v.to(device) for k, v in data_dict.items()}
        # if USE_GENERATIVE_TRAINING:
        pcpt_gene = data_dict["pcpt_ids"]
        pcpt_expr = data_dict["pcpt_values"]
        pcpt_key_padding_mask = pcpt_gene.eq(vocab.pad_index)
        gen_gene = data_dict["gen_ids"]
        gen_expr_target = target_values = data_dict["gen_values"]
        gen_key_padding_mask = gen_gene.eq(vocab.pad_index)
        # else:
        #     input_gene_ids = data_dict["gene"]
        #     input_values = data_dict["masked_expr"]
        #     target_values = data_dict["expr"]
        #     src_key_padding_mask = input_gene_ids.eq(vocab[args.pad_token])

        with torch.amp.autocast(device, enabled=enable_fp16):
            # if USE_GENERATIVE_TRAINING:
            output_dict = model(
                pcpt_gene,
                pcpt_expr,
                pcpt_key_padding_mask,
                gen_gene,
                gen_key_padding_mask,
                # CLS=use_cls,
                # MVC=use_mvc,
                # generative_training=True,
            )
            gen_expr_preds = output_values = output_dict["gen_preds"]

            positions_to_match = ~gen_key_padding_mask
            loss = loss_mse = masked_mse_loss(
                gen_expr_preds, gen_expr_target, positions_to_match
            )
            # wandb.log({"train/mse": loss_mse.item()}, step=global_iter)

            # if use_mvc:
            #     loss_mvc = criterion(
            #         output_dict["mvc_output"][:, pcpt_gene.shape[1] :],
            #         gen_expr_target,
            #         positions_to_match,
            #     )
            #     loss = loss + loss_mvc
            #     writer.add_scalar("train/mvc", loss_mvc, global_iter)
            # else:
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
            #     if USE_CCE:
            #         loss_cce = 10 * output_dict["loss_cce"]
            #         loss = loss + loss_cce
            #         writer.add_scalar("train/cce", loss_cce, global_iter)
            #     if MVC:
            #         loss_mvc = criterion(
            #             output_dict["mvc_output"], target_values, positions_to_match
            #         )
            #         loss = loss + loss_mvc
            #         writer.add_scalar("train/mvc", loss_mvc, global_iter)

            wandb.log({"train/loss_pcpt": loss.item()}, step=global_iter)

            # if USE_GENERATIVE_TRAINING and global_iter > 1000:
            previous_cell_embs = output_dict["cell_emb"].detach()
            preds = model(
                pcpt_gene,
                pcpt_expr,
                pcpt_key_padding_mask,
                gen_gene,
                gen_key_padding_mask,
                # CLS=False,
                # MVC=False,
                input_cell_emb=previous_cell_embs,
                # generative_training=True,
            )["gen_preds"]
            loss_gen = masked_mse_loss(preds, gen_expr_target, positions_to_match)
            loss = loss + loss_gen
            wandb.log({"train/loss_gen": loss_gen.item()}, step=global_iter)

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

        if grad_accu_steps > 1:
            loss = loss / grad_accu_steps
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if grad_accu_steps > 1:
            if batch % grad_accu_steps == 0 or batch == num_batches - 1:
                scheduler.step()
                optimizer.zero_grad()
        else:
            scheduler.step()
            optimizer.zero_grad()

        with torch.no_grad():
            mre = masked_relative_error(
                output_values, target_values, positions_to_match
            )
            wandb.log({"train/mre": mre.item()}, step=global_iter)

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
            ms_per_batch = (time.time() - epoch_start_time) * 1000 / log_interval
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

            wandb.log({
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
            save_dir=save_dir,
            logger=logger,
            vocab=vocab,
            device=device,
            best_val_loss=best_val_loss,
            enable_fp16=enable_fp16,
            global_iter=global_iter,
            # save=(save_interval > 0 and batch % save_interval == 0),
        )

        best_val_loss = min(best_val_loss, val_loss)

        model.train()  # important, reset to train mode
        val_losses.append(val_loss)
        val_mres.append(val_mre)

    epoch_val_loss = np.mean(val_losses)
    epoch_val_mre = np.mean(val_mres)

    return epoch_val_loss, epoch_val_mre

# we need to be careful when saving checkpoints since preemption can also
# occur during checkpointing. Therefore, we need to make sure the checkpoint
# file is either kept untouched or successfully updated during this process.
def commit_state(model, optimizer, scheduler, grad_scaler, rng, cuda_rng, epoch, best_val_loss, patience_counter, checkpoint_path):

    temp_path = os.path.join(os.path.dirname(checkpoint_path), "temp.pt")

    training_state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "grad_scaler": grad_scaler.state_dict() if grad_scaler else None,
        "rng": rng,
        "cuda_rng": cuda_rng,
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "patience_counter": patience_counter,
    }

    # first save to temp file
    torch.save(training_state, temp_path)
    # according to the GNU spec of rename, the state of checkpoint_path
    # is atomic, i.e. it will either be modified or not modified, but not in
    # between, during a system crash (i.e. preemtion)
    os.replace(temp_path, checkpoint_path)
    logger.info("Training state committed to {} at time {}".format(checkpoint_path, time.ctime(time.time())))

def create_or_restore_data_state_and_wandb(hmc_table_path, taxa_path, wandb_enabled, wandb_entity, wandb_project, init_lr, batch_size, max_epochs, cosine_warmup_ratio, binning, data_restore_path, nrows=None):
    if os.path.exists(data_restore_path):
        # load the data state from the file
        with open(data_restore_path, 'rb') as f:
            data_state = torch.load(f, weights_only=False)
        logger.info("Data state restored from {}".format(data_restore_path))
        train_data_dict = data_state["train_data_dict"]
        valid_data_dict = data_state["valid_data_dict"]
        vocab = data_state["vocab"]

        run = wandb.init(
            id=data_state["wandb_run_id"],
            resume="must",
            mode="online" if wandb_enabled else "disabled",
            entity=wandb_entity,
            config={
                "learning_rate": init_lr,
                "batch_size": batch_size,
                "max_epochs": max_epochs,
                "cosine_warmup_ratio": cosine_warmup_ratio,
                "binning": binning
            },
            project=wandb_project,
        )

    else:
        run = wandb.init(
            mode="online" if wandb_enabled else "disabled",
            entity=wandb_entity,
            project=wandb_project,
            config={
                "learning_rate": init_lr,
                "batch_size": batch_size,
                "max_epochs": max_epochs,
                "cosine_warmup_ratio": cosine_warmup_ratio,
                "binning": binning
            },
            resume="allow"
        )

        if nrows:
            hmc_npy = np.load(hmc_table_path)[:nrows,:,:] # shape (num_samples, num_taxa, 2) where (:,:,0) is taxa_id and (:,:,1) is counts
        else:
            hmc_npy = np.load(hmc_table_path)

        with open(taxa_path, "r") as f:
            taxa_list = json.load(f)

        preprocessor = Preprocessor(
            binning=wandb.config["binning"],
        )

        _, _ = preprocessor.process_from_np(hmc_npy)

        vocab = MicrobiomeVocab(taxa_list)  # Replace with your vocab

        # create tokenizer
        tokenizer = Tokenizer(vocab)
        data_dict = tokenizer.tokenize_and_pad_batch(hmc_npy)
        # Assuming data_dict is a dictionary with keys 'taxa_ids' and 'values'

        # train and validation split
        (
            train_taxa_ids,
            valid_taxa_ids,
            train_values,
            valid_values
        ) = train_test_split(
            data_dict["taxa_ids"],
            data_dict["values"],
            test_size=0.2,
            shuffle=True
        )

        train_data_dict = {
            "taxa_ids": train_taxa_ids,
            "values": train_values
        }
        valid_data_dict = {
            "taxa_ids": valid_taxa_ids,
            "values": valid_values
        }

        # save the train and validation dataloaders
        data_state = {
            "train_data_dict": train_data_dict,
            "valid_data_dict": valid_data_dict,
            "vocab": vocab,
            "wandb_run_id": run.id if wandb_enabled else None,
        }

        # save the data state to the file
        torch.save(data_state, data_restore_path)
        logger.info("Data state saved to {}".format(data_restore_path))

    return train_data_dict, valid_data_dict, vocab, run


def create_or_restore_training_state(vocab, init_lr, warmup_ratio_or_step, total_epochs, trainloader_length, device, checkpoint_path):
    # initial configuration of the model
    model = TransformerModel(
        d_model=512,
        nhead=8,
        d_hid=2048,
        nlayers=6,
        vocab=vocab,
        dropout=0.1,
        use_generative_training=True,
    )

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
    # setup scheduler
    # if warmup_ratio_or_step > 0:
    assert warmup_ratio_or_step >= 0, "Warmup ratio or step must be non-negative"
    total_num_batches = trainloader_length * total_epochs

    warmup_steps = (
        int(total_num_batches * warmup_ratio_or_step)
        if warmup_ratio_or_step < 1
        else int(warmup_ratio_or_step)
    )
    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_num_batches,
    )

    # else:
    #     scheduler = torch.optim.lr_scheduler.StepLR(
    #         optimizer, scheduler_interval, gamma=args.scheduler_factor
    #     )
    scaler = torch.amp.GradScaler(device)
    # dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size,
    #                         sampler=StatefulSampler(dataset, shuffle=True),
    #                         num_workers=0)
    epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0
    # restore training state if checkpoint exists
    if os.path.exists(checkpoint_path):
        training_state = torch.load(checkpoint_path, weights_only=False)

        model.load_state_dict(training_state['model'])
        optimizer.load_state_dict(training_state['optimizer'])
        scheduler.load_state_dict(training_state['scheduler'])
        scaler.load_state_dict(training_state['grad_scaler'])
        rng = training_state['rng']
        torch.random.set_rng_state(rng)
        cuda_rng = training_state['cuda_rng']
        if cuda_rng is not None and device == 'cuda':
            torch.cuda.set_rng_state(cuda_rng)
        epoch = training_state['epoch']
        best_val_loss = training_state['best_val_loss']
        patience_counter = training_state['patience_counter']
        logger.info(f"training state restored at beginning of epoch {epoch + 1}")

    else:
        logger.info("No checkpoint detected, starting from initial state")

    return model, optimizer, scaler, scheduler, epoch, best_val_loss, patience_counter



def eval_and_save(
    model: nn.Module,
    valid_loader: DataLoader,
    save_dir: str,
    logger,
    vocab: MicrobiomeVocab,
    device: str,
    best_val_loss: float,
    global_iter: int,
    enable_fp16: bool = False,
    # save: bool = True,
) -> None:
    # perform evaluation in distributed data parallel
    val_loss, val_mre = evaluate(model, valid_loader, vocab, enable_fp16, device).values()
    # if IS_DATA_PARALLEL:
    #     # gather the results from all the processes
    #     val_loss_list = [torch.zeros_like(val_loss) for _ in range(world_size)]
    #     val_mre_list = [torch.zeros_like(val_mre) for _ in range(world_size)]
    #     torch.distributed.all_gather(val_loss_list, val_loss)
    #     torch.distributed.all_gather(val_mre_list, val_mre)
    #     val_loss = torch.mean(torch.stack(val_loss_list))
    #     val_mre = torch.mean(torch.stack(val_mre_list))
    val_loss, val_mre = val_loss.item(), val_mre.item()

    logger.info(f"valid loss/mse {val_loss:5.4f} | mre {val_mre:5.4f}")
    wandb.log({
        "val/val_loss": val_loss,
        "val/val_mre": val_mre,
    }, step=global_iter)

    if val_loss < best_val_loss:
        # save the best model
        logger.info(f"Saving the best model to {save_dir}")
        torch.save(
            # model.module.state_dict()
            # if isinstance(
            #     model, (nn.DataParallel, nn.parallel.DistributedDataParallel)
            # )
            # else model.state_dict(),
            model.state_dict(),
            save_dir + "/best_model.pt",
        )

    # if save:
    #     torch.save(
    #         # model.module.state_dict()
    #         # if isinstance(
    #         #     model, (nn.DataParallel, nn.parallel.DistributedDataParallel)
    #         # )
    #         # else model.state_dict(),
    #         model.state_dict(),
    #         save_dir + f"/model-{'ep' if is_epoch else ''}{iter_or_epoch}.pt",
    #     )

    # if IS_DATA_PARALLEL:
    #     torch.distributed.barrier()

    return val_loss, val_mre


def epoch_end_logs(epoch_start_time, epoch, val_loss, val_mre):
    elapsed = time.time() - epoch_start_time
    logger.info("-" * 89)
    logger.info(
        f"| end of epoch {epoch + 1:3d} | time: {elapsed:5.2f}s | "
        f"valid loss/mse {val_loss:5.4f} | mre {val_mre:5.4f}"
    )
    logger.info(f"{'-' * 89}\n")
    # writer.add_scalar("valid/mse", val_loss, iter_or_epoch * len(valid_loader))
    # writer.add_scalar("valid/mre", val_mre, iter_or_epoch * len(valid_loader))

def evaluate(
        model: nn.Module,
        valid_loader: DataLoader,
        vocab: MicrobiomeVocab,
        enable_fp16: bool,
        device: str,
        ) -> Dict[str, torch.Tensor]:
    """
    Evaluate the model on the evaluation data.
    """
    model.eval()
    total_loss = 0.0
    total_error = 0.0
    with torch.no_grad():
        for data_dict in valid_loader:

            data_dict = {k: v.to(device) for k, v in data_dict.items()}
            # if USE_GENERATIVE_TRAINING:
            pcpt_ids = data_dict["pcpt_ids"]
            pcpt_values = data_dict["pcpt_values"]
            pcpt_key_padding_mask = pcpt_ids.eq(vocab.pad_index)
            gen_ids = data_dict["gen_ids"]
            gen_values = data_dict["gen_values"]
            gen_key_padding_mask = gen_ids.eq(vocab.pad_index)
            # else:
            #     input_gene_ids = data_dict["gene"]
            #     input_values = data_dict["masked_expr"]
            #     target_values = data_dict["expr"]
            #     src_key_padding_mask = input_gene_ids.eq(vocab[args.pad_token])

            with torch.amp.autocast(device, enabled=enable_fp16):
                # if USE_GENERATIVE_TRAINING:
                output_dict = model(
                    pcpt_ids,
                    pcpt_values,
                    pcpt_key_padding_mask,
                    gen_ids,
                    gen_key_padding_mask,
                    # CLS=False,
                    # MVC=False,
                    # generative_training=True,
                )
                gen_expr_preds = output_values = output_dict["gen_preds"]

                positions_to_match = ~gen_key_padding_mask
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

                loss = masked_mse_loss(output_values, gen_values, positions_to_match)
            total_loss += loss.item()
            total_error += masked_relative_error(
                output_values, gen_values, positions_to_match
            ).item()

    total_loss = total_loss / len(valid_loader)
    total_error = total_error / len(valid_loader)
    return {
        "mse": torch.tensor(total_loss, device=device, dtype=torch.float),
        "mre": torch.tensor(total_error, device=device, dtype=torch.float),
    }
