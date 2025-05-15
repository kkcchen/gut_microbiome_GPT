"""
original code from
Learning a deep language model for microbiomes: the power of large scale unlabeled microbiome data
Quintin Pope, Rohan Varma, Chritine Tataru, Maude David, Xiaoli Fern
bioRxiv 2023.07.17.549267; doi: https://doi.org/10.1101/2023.07.17.549267

edited by Haoze Deng
"""
import os
import pathlib

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim import SGD
from torch.utils.data import DataLoader
from models import ElectraGenerator
import tqdm
import pdb
import wandb
import logging


class GeneratorTrainer:
    """
    ELECTRATrainer make the pretrained ELECTRA model
    """

    def __init__(self, electra: ElectraGenerator, vocab_size: int,
                 train_dataloader: DataLoader, test_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 100, log_file=None, append=False,
                 wandb_run=None):
        """
        :param electra: ELECTRA model which you want to train
        :param vocab_size: total word vocab size
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param with_cuda: traning with cuda
        :param log_freq: logging frequency of the batch iteration
        """
        # Setup cuda device for ELECTRA training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")
        self.hardware = "cuda" if cuda_condition else "cpu"

        # This ELECTRA model will be saved every epoch
        self.electra = electra.to(self.device)
        self.electra = self.electra.float()

        # pdb.set_trace()
        # Distributed GPU training if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for ELECTRA" % torch.cuda.device_count())
            self.electra = nn.DataParallel(self.electra, device_ids=cuda_devices)
            self.hardware = "parallel"

        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        # Setting the Adam optimizer with hyper-param
        # self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        # self.optim_schedule = ScheduledOptim(self.optim, self.electra.hidden, n_warmup_steps=warmup_steps)
        self.optim = SGD(self.electra.parameters(), lr=lr, momentum=0.9)
        # self.optim = Adam(self.electra.parameters(),lr=lr,eps=1e-06)

        self.log_freq = log_freq

        # clear log file
        if log_file:
            self.log_file = log_file
            if not append:
                # with open(self.log_file,"w+") as f:
                #     f.write("EPOCH,MODE,AVG LOSS,TOTAL CORRECT,TOTAL ELEMENTS,ACCURACY,MASK CORRECT,MASK 3 CORRECT,MASK 5 CORRECT,MASK 10 CORRECT,TOTAL MASK,MASK ACCURACY\n")
                ...
        print("Total Parameters:", sum([p.nelement() for p in self.electra.parameters()]))

        self.wandb_run = wandb_run
        self.logger = self._setup_logger()

    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch, data_loader, train=True):
        """
        Loop over data_loader for training or evaluation.
        Logs metrics at each iteration and at epoch end to Weights & Biases (wandb).

        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader
        :param train: whether to perform backprop (True for training)
        :return: None
        """
        mode = "train" if train else "test"
        total_steps = len(data_loader)

        # Initialize accumulators
        epoch_total_loss = 0.0
        epoch_token_correct = 0
        epoch_token_total = 0
        epoch_mask_total = 0
        epoch_mask_correct = {1: 0, 3: 0, 5: 0, 10: 0}

        # Progress bar
        data_iter = tqdm.tqdm(
            enumerate(data_loader),
            desc=f"EP_{mode}:{epoch}",
            total=total_steps
        )

        for i, batch in data_iter:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Build mask for frequencies == 0
            zero_mask = batch["species_frequencies"].eq(0)
            attn_mask = (~zero_mask).float()

            # Prepare labels: ignore non-mask tokens
            masked_labels = batch["electra_label"].masked_fill(~batch["mask_locations"], -100)

            # Forward pass
            g_loss, g_scores = self.electra.forward(batch["electra_input"], attn_mask, masked_labels)

            # Backward & optimize
            if train:
                self.optim.zero_grad()
                loss_to_backprop = g_loss.mean() if self.hardware == "parallel" else g_loss
                loss_to_backprop.backward()
                self.optim.step()

            # Convert to scalars
            iter_loss = g_loss.mean().item() if self.hardware == "parallel" else g_loss.item()
            epoch_total_loss += iter_loss

            # Predictions: (batch_size, seq_len, vocab_size)
            preds = g_scores.sort(dim=-1, descending=True)[1]

            # Masked positions predictions & labels
            mask_positions = batch["mask_locations"]
            num_mask = mask_positions.sum().item()
            flat_preds = preds[mask_positions]
            flat_labels = masked_labels[mask_positions]

            # Token-level accuracy (all tokens)
            num_token = attn_mask.sum().item()
            correct_tokens = (preds[..., 0][attn_mask.bool()] == batch["electra_label"][attn_mask.bool()]).sum().item()

            # Mask-level correct counts for various top-k
            correct_top = {}
            for k in epoch_mask_correct.keys():
                correct_top[k] = GeneratorTrainer.check_top_x_mask_predictions(flat_preds, flat_labels, k)

            # Update accumulators
            epoch_token_correct += correct_tokens
            epoch_token_total += num_token
            epoch_mask_total += num_mask
            for k, cnt in correct_top.items():
                epoch_mask_correct[k] += cnt

            # Iteration metrics
            iter_mask_acc = {f"mask_acc_top{k}": (correct_top[k] / num_mask if num_mask else 0) for k in correct_top}
            iter_token_acc = correct_tokens / num_token if num_token > 0 else 0

            # Log per-iteration to wandb
            # if train and i % self.log_freq == 0:
            #     metrics = {
            #         f"{mode}/iter/loss": iter_loss,
            #         **{f"{mode}/iter/{m}": v for m, v in iter_mask_acc.items()},
            #         f"{mode}/iter/token_acc": iter_token_acc,
            #     }
            #     if self.wandb_run is not None:
            #         self.wandb_run.log(metrics, step=epoch * total_steps + i)
            #     # self.logger.info(
            #     #     "Iter %d | loss=%.4f | mask_acc_top1=%.4f | token_acc=%.4f",
            #     #     i,
            #     #     iter_loss,
            #     #     iter_mask_acc["mask_acc_top1"],
            #     #     iter_token_acc,
            #     # )
            data_iter.set_postfix(
                loss=f"{iter_loss:.4f}",
                mask_acc=f"{iter_mask_acc['mask_acc_top1']:.4f}",
                token_acc=f"{iter_token_acc:.4f}"
            )

            # Clean up
            del batch, attn_mask, masked_labels, g_loss, g_scores, preds

        # Epoch-level metrics
        avg_loss = epoch_total_loss / total_steps
        epoch_mask_acc = {f"mask_acc_top{k}": (epoch_mask_correct[k] / epoch_mask_total if epoch_mask_total else 0)
                          for k in epoch_mask_correct}
        epoch_token_acc = epoch_token_correct / epoch_token_total if epoch_token_total else 0

        # Log epoch metrics
        epoch_metrics = {
            "epoch": epoch,
            f"{mode}/epoch/loss": avg_loss,
            **{f"{mode}/epoch/{m}": v for m, v in epoch_mask_acc.items()},
            f"{mode}/epoch/token_acc": epoch_token_acc,
        }
        if self.wandb_run is not None:
            # temporary
            if mode == "train":
                step = epoch * 2
            else:
                step = (epoch * 2) + 1
            self.wandb_run.log(epoch_metrics, step=step)
        self.logger.info(
            "Epoch %d [%s] | avg_loss=%.4f | mask_acc_top1=%.2f%% | token_acc=%.2f%%",
            epoch,
            mode,
            avg_loss,
            epoch_mask_acc["mask_acc_top1"] * 100,
            epoch_token_acc * 100,
        )
        # Print or save
        if not train:
            self.logger.info(
                "Logged test results to %s: loss=%.4f, token_acc=%.2f%%, mask_acc_top1=%.2f%%",
                self.log_file,
                avg_loss,
                epoch_token_acc * 100,
                epoch_mask_acc["mask_acc_top1"] * 100,
            )

    @staticmethod
    def check_top_x_mask_predictions(preds, labels, x):
        """
        Count how many times the true label is within the top-x predictions.
        preds: Tensor of shape (num_masks, vocab_size) sorted descending by score
        labels: Tensor of shape (num_masks,)
        """
        top_x = preds[:, :x]
        return (top_x == labels.unsqueeze(1)).any(dim=1).sum().item()

    @staticmethod
    def _setup_logger():
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "[%(asctime)s] %(levelname)s:%(name)s: %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger

    def save(self, epoch, file_path):
        """
        Saving the current ELECTRA model on file_path

        :param epoch: current epoch number
        :param file_path: model output directory
        """
        # output_file_path = file_path + "_epoch{}".format(epoch)
        output_file_path = os.path.join(file_path, f"epoch_{epoch}")
        os.mkdir(output_file_path) if not os.path.exists(output_file_path) else 0
        if self.hardware == "parallel":
            # pdb.set_trace()
            # self.electra.module.generator.save_pretrained(output_file_path + "_gen")
            self.electra.module.generator.save_pretrained(os.path.join(output_file_path, "generator_weights"))
            # torch.save(self.electra.module.embed_layer.state_dict(), output_file_path + "_gen_embed")
            torch.save(self.electra.module.embed_layer.state_dict(), os.path.join(output_file_path, "embed_layer_weights"))
        else:
            # self.electra.generator.save_pretrained(output_file_path + "_gen")
            # torch.save(self.electra.embed_layer.state_dict(), output_file_path + "_gen_embed")
            self.electra.generator.save_pretrained(os.path.join(output_file_path, "generator_weights"))
            torch.save(self.electra.embed_layer.state_dict(), os.path.join(output_file_path, "embed_layer_weights"))

    def save_slurm(self, epoch, file_path):
        """
        Saving the current ELECTRA model on file_path for slurm
        """
        gen_embed_save_path = os.path.join(file_path, "gen_embed")
        gen_model_save_path = os.path.join(file_path, "gen_model")
        resume_from_save_path = os.path.join(file_path, "resume_from.txt")

        if self.hardware == "parallel":
            self.electra.module.generator.save_pretrained(gen_model_save_path)
            torch.save(self.electra.module.embed_layer.state_dict(), gen_embed_save_path)
        else:
            self.electra.generator.save_pretrained(gen_model_save_path)
            torch.save(self.electra.embed_layer.state_dict(), gen_embed_save_path)

        resume_from_save_path = pathlib.Path(resume_from_save_path)
        resume_from_save_path.write_text(str(epoch + 1))
