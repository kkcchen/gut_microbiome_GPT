"""
original code from
Learning a deep language model for microbiomes: the power of large scale unlabeled microbiome data
Quintin Pope, Rohan Varma, Chritine Tataru, Maude David, Xiaoli Fern
bioRxiv 2023.07.17.549267; doi: https://doi.org/10.1101/2023.07.17.549267

edited by Haoze Deng
"""
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim import SGD
from torch.utils.data import DataLoader

from models import ElectraGenerator, ElectraDiscriminator
import tqdm
import pdb
import logging
import os
import pathlib


class DiscriminatorTrainer:
    """
    ELECTRATrainer make the pretrained ELECTRA model
    """

    def __init__(self, electra_gen: ElectraGenerator, electra_disc: ElectraDiscriminator, vocab_size: int,
                 train_dataloader: DataLoader, test_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 100, d_log_file=None, append=False,
                 freeze_embed=False, wandb_run=None):
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

        print(freeze_embed)
        # Setup cuda device for ELECTRA training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")
        self.hardware = "cuda" if cuda_condition else "cpu"

        # This ELECTRA model will be saved every epoch
        self.electra_gen = electra_gen.to(self.device)
        self.electra_gen = self.electra_gen.float()

        self.electra_disc = electra_disc.to(self.device)
        self.electra_disc = self.electra_disc.float()

        # pdb.set_trace()
        # Distributed GPU training if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for ELECTRA" % torch.cuda.device_count())
            self.electra_gen = nn.DataParallel(self.electra_gen, device_ids=cuda_devices)
            self.electra_disc = nn.DataParallel(self.electra_disc, device_ids=cuda_devices)
            self.hardware = "parallel"

        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        # Setting the Adam optimizer with hyper-param
        # self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        # self.optim_schedule = ScheduledOptim(self.optim, self.electra.hidden, n_warmup_steps=warmup_steps)
        # self.optimd = Adam(self.electra_disc.parameters(),lr=lr,eps=1e-06)
        if freeze_embed:
            if self.hardware == "parallel":
                self.electra_disc.module.embed_layer.weight.requires_grad = False
            else:
                self.electra_disc.embed_layer.weight.requires_grad = False
        self.optimd = SGD([param for param in self.electra_disc.parameters() if param.requires_grad == True], lr=lr,
                          momentum=0.9)

        self.log_freq = log_freq

        # clear log file
        if d_log_file:
            self.d_log_file = d_log_file
            if not append:
                # with open(self.d_log_file, "w+") as f:
                #     f.write(
                #         "EPOCH,MODE,AVG LOSS,TOTAL CORRECT,TOTAL ELEMENTS,ACCURACY,MASK CORRECT,TOTAL MASK,MASK ACCURACY\n")
                ...

        self.wandb_run = wandb_run
        self.logger = self._setup_logger()

    def swap_gen(self, new_gen, with_cuda=True, cuda_devices=None):
        del self.electra_gen
        self.electra_gen = new_gen.to(self.device)
        self.electra_gen.float()
        if with_cuda and torch.cuda.device_count() > 1:
            self.electra_gen = nn.DataParallel(self.electra_gen, device_ids=cuda_devices)

    def train(self, epoch, disc=False):
        self.iteration(epoch, self.train_data, train=True)

    def test(self, epoch, disc=False):
        self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch, data_loader, train=True):
        """
        loop over the data_loader for training or testing
        if on train status, backward operation is activated
        and also auto save the model every peoch

        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value of is train or test
        :return: None
        """
        mode = "train" if train else "test"
        total_steps = len(data_loader)
        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(
            enumerate(data_loader),
            desc=f"EP_{mode}:{epoch}",
            total=total_steps
        )

        epoch_total_loss = 0.0
        epoch_total_correct = 0
        epoch_total_element = 0
        epoch_mask_correct = 0
        epoch_total_mask = 0
        for i, data in data_iter:

            # 0. batch_data will be sent into the device(GPU or cpu)
            data = {key: value.to(self.device) for key, value in data.items()}

            # create attention mask
            zero_boolean = torch.eq(data["species_frequencies"], 0)
            mask = torch.ones(zero_boolean.shape, dtype=torch.float).to(self.device)
            mask = mask.masked_fill(zero_boolean, 0)

            # change label for non-masked tokens to -100 so generator ignores predictions on non-masked tokens
            data["electra_mask_label"] = data["electra_label"].masked_fill(~data["mask_locations"], -100)

            self.electra_gen.eval()
            with torch.no_grad():
                g_loss, g_scores = self.electra_gen(data['electra_input'], mask, data['electra_mask_label'])
                g_predictions = g_scores.max(2)[1]
                disc_labels = (data["electra_label"] != g_predictions).long()
                disc_labels = disc_labels.masked_fill(~data["mask_locations"], 0)
                disc_inputs = torch.where(data["mask_locations"], g_predictions, data["electra_input"])
            if not train:
                self.electra_disc.eval()
                with torch.no_grad():
                    d_loss, d_scores = self.electra_disc(disc_inputs, mask, disc_labels)
            else:
                self.electra_disc.train()
                d_loss, d_scores = self.electra_disc(disc_inputs, mask, disc_labels)

            del g_predictions
            del disc_inputs

            # 3. backward and optimization only in train
            if train:
                # self.optim_schedule.zero_grad()
                self.optimd.zero_grad()
                if self.hardware == "parallel":
                    d_loss.mean().backward()
                else:
                    d_loss.backward()
                # self.optim_schedule.step_and_update_lr()
                self.optimd.step()

            del g_scores

            # get discriminator accuracy for all tokens
            d_predictions = torch.where(d_scores > 0.5, torch.tensor([1]).to(self.device),
                                        torch.tensor([0]).to(self.device))
            d_mask_predictions = torch.masked_select(d_predictions, data["mask_locations"])
            d_mask_token_labels = torch.masked_select(disc_labels, data["mask_locations"])
            iter_mask_correct = torch.sum(d_mask_predictions == d_mask_token_labels).item()
            iter_total_mask = d_mask_token_labels.shape[0]
            epoch_mask_correct += iter_mask_correct
            epoch_total_mask += iter_total_mask

            del d_mask_predictions
            del d_mask_token_labels

            # get discriminator accuracy for all tokens
            iter_total_correct = torch.masked_select((d_predictions == disc_labels), mask.bool()).sum().item()
            iter_total_element = mask.sum().item()
            epoch_total_correct += iter_total_correct
            epoch_total_element += iter_total_element

            iter_loss = 0
            if self.hardware == "parallel":
                epoch_total_loss += d_loss.sum().item()
                iter_loss = d_loss.sum().item()

            else:
                epoch_total_loss += d_loss.item()
                iter_loss = d_loss.item()

            # if i % self.log_freq == 0 and total_mask > 0:
            #     # pdb.set_trace()
            #     data_iter.write(
            #         "epoch: {}, iter: {}, avg loss: {},accuracy: {}/{}={:.2f}%, mask accuracy: {}/{}={:.2f}%, loss: {}".format(
            #             epoch, i, cumulative_loss / ((i + 1) * data_loader.batch_size), d_total_correct, total_element,
            #             d_total_correct / total_element * 100, d_total_mask_correct, total_mask,
            #             d_total_mask_correct / total_mask * 100, log_loss))
            iter_mask_acc = iter_mask_correct / iter_total_mask if iter_mask_correct > 0 else 0
            iter_total_acc = iter_total_correct / iter_total_element if iter_total_element > 0 else 0
            data_iter.set_postfix(
                loss=f"{iter_loss:.4f}",
                mask_acc=f"{iter_mask_acc:.4f}",
                token_acc=f"{iter_total_acc:.4f}"
            )
            del data
            del mask
            del d_scores
            del disc_labels
            del d_predictions
            del d_loss
            del g_loss

        # Epoch-level metrics
        avg_loss = epoch_total_loss / total_steps
        epoch_mask_acc = epoch_mask_correct / epoch_total_mask if epoch_total_mask > 0 else 0
        epoch_token_acc = epoch_total_correct / epoch_total_element if epoch_total_element > 0 else 0

        # Log epoch metrics
        epoch_metrics = {
            "epoch": epoch,
            f"{mode}/epoch/loss": avg_loss,
            f"{mode}/epoch/mask_acc": epoch_mask_acc,
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
            epoch_mask_acc * 100,
            epoch_token_acc * 100,
        )
        # Print or save
        if not train:
            self.logger.info(
                "Logged test results to %s: loss=%.4f, token_acc=%.2f%%, mask_acc_top1=%.2f%%",
                self.d_log_file,
                avg_loss,
                epoch_token_acc * 100,
                epoch_mask_acc * 100,
            )
        # print("EP{}_{}, avg_loss={}, accuracy={:.2f}%".format(epoch, str_code, cumulative_loss / (
        #             len(data_iter) * data_loader.batch_size), d_total_mask_correct / total_mask * 100))
        # if self.d_log_file:
        #     f = open(self.d_log_file, "a")
        #     f.write("{},{},{},{},{},{},{},{},{}\n".format(epoch, str_code,
        #                                                   cumulative_loss / (len(data_loader) * data_loader.batch_size),
        #                                                   d_total_correct, total_element,
        #                                                   d_total_correct / total_element * 100, d_total_mask_correct,
        #                                                   total_mask, d_total_mask_correct / total_mask * 100))

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
            self.electra_disc.module.discriminator.save_pretrained(os.path.join(output_file_path, "discriminator_weights"))
            # torch.save(self.electra.module.embed_layer.state_dict(), output_file_path + "_gen_embed")
            torch.save(self.electra_disc.module.embed_layer.state_dict(), os.path.join(output_file_path, "embed_layer_weights"))
        else:
            # self.electra.generator.save_pretrained(output_file_path + "_gen")
            # torch.save(self.electra.embed_layer.state_dict(), output_file_path + "_gen_embed")
            self.electra_disc.discriminator.save_pretrained(os.path.join(output_file_path, "discriminator_weights"))
            torch.save(self.electra_disc.embed_layer.state_dict(), os.path.join(output_file_path, "embed_layer_weights"))

    def save_slurm(self, epoch, generator_id, file_path):
        """
        Saving the current ELECTRA model on file_path for slurm
        """
        gen_embed_save_path = os.path.join(file_path, "gen_embed")
        gen_model_save_path = os.path.join(file_path, "gen_model")
        resume_from_save_path = os.path.join(file_path, "resume_from.txt")

        if self.hardware == "parallel":
            self.electra_disc.module.discriminator.save_pretrained(gen_model_save_path)
            torch.save(self.electra_disc.module.embed_layer.state_dict(), gen_embed_save_path)
        else:
            self.electra_disc.discriminator.save_pretrained(gen_model_save_path)
            torch.save(self.electra_disc.embed_layer.state_dict(), gen_embed_save_path)

        resume_from_save_path = pathlib.Path(resume_from_save_path)
        resume_from_save_path.write_text(str(f"{generator_id},{epoch}"))


