import time
from typing import List, Dict, Any
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
# import wandb

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
        save_interval: int,
        save_dir: str,
        device: str,
        logger,
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
    start_time = time.time()

    num_batches = len(train_loader)
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
            # writer.add_scalar("train/mse", loss_mse, global_iter)
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



            # convert to wandb here too!!!
            # writer.add_scalar("train/loss", loss, global_iter)

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
                # writer.add_scalar("train/gen", loss_gen, global_iter)

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
            # writer.add_scalar("train/mre", mre, global_iter)

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
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            cur_mse = total_mse / log_interval
            # cur_cls = total_cls / log_interval if USE_CLS else 0.0
            cur_gen = total_gen / log_interval if "loss_gen" in locals() else 0.0
            # cur_mvc = total_mvc / log_interval if MVC else 0.0
            cur_error = total_error / log_interval
            # ppl = math.exp(cur_loss)
            logger.info(
                f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {cur_loss:5.2f} | mse {cur_mse:5.2f} | mre {cur_error:5.2f} |"
                # + (f"cls {cur_cls:5.2f} | " if USE_CLS else "")
                + (f"gen {cur_gen:5.2f} |" if "loss_gen" in locals() else "")
                # + (f"mvc {cur_mvc:5.2f} |" if MVC else "")
            )
            # writer.add_scalar("lr", lr, global_iter)

            total_loss = 0
            total_mse = 0
            # total_cls = 0
            total_gen = 0
            # total_mvc = 0
            total_error = 0
            start_time = time.time()

        # immediately eval and save
        # if batch % save_interval == 0 and batch > 0:

        best_val_loss = eval_and_save(
            model=model,
            valid_loader=valid_loader,
            iter_or_epoch=global_iter,
            save_dir=save_dir,
            epoch_start_time=start_time,
            logger=logger,
            vocab=vocab,
            device=device,
            best_val_loss=best_val_loss,
            enable_fp16=enable_fp16,
            is_epoch=False,
            save=True,
        )
        model.train()  # important, reset to train mode

        return best_val_loss


def eval_and_save(
    model: nn.Module,
    valid_loader: DataLoader,
    iter_or_epoch: int,
    save_dir: str,
    epoch_start_time: float,
    logger,
    vocab: MicrobiomeVocab,
    device: str,
    best_val_loss: float,
    enable_fp16: bool = False,
    is_epoch: bool = False,
    save: bool = True,
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

    # if args.local_rank in [0, -1]:
    if is_epoch:
        elapsed = time.time() - epoch_start_time
        logger.info("-" * 89)
        logger.info(
            f"| end of epoch {iter_or_epoch:3d} | time: {elapsed:5.2f}s | "
            f"valid loss/mse {val_loss:5.4f} | mre {val_mre:5.4f}"
        )
        logger.info(f"{'-' * 89}\n")
        # writer.add_scalar("valid/mse", val_loss, iter_or_epoch * len(valid_loader))
        # writer.add_scalar("valid/mre", val_mre, iter_or_epoch * len(valid_loader))
    else:
        logger.info(f"valid loss/mse {val_loss:5.4f} | mre {val_mre:5.4f}")
        # writer.add_scalar("valid/mse", val_loss, iter_or_epoch)
        # writer.add_scalar("valid/mre", val_mre, iter_or_epoch)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
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

    if save:
        torch.save(
            # model.module.state_dict()
            # if isinstance(
            #     model, (nn.DataParallel, nn.parallel.DistributedDataParallel)
            # )
            # else model.state_dict(),
            model.state_dict(),
            save_dir + f"/model-{'ep' if is_epoch else ''}{iter_or_epoch}.pt",
        )

    # if IS_DATA_PARALLEL:
    #     torch.distributed.barrier()

    return best_val_loss



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
