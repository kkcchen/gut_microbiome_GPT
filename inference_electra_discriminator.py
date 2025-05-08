import argparse
import pdb
from torch.utils.data import DataLoader

import torch
# from trainers import DiscriminatorTrainer
from data_utils import ELECTRADataset
from transformers import ElectraConfig
from models import ElectraDiscriminator
# from models import ElectraGenerator
from models import ElectraFineTuner
from trainers import DiscriminatorFinetuneTrainer
import numpy as np
from tqdm import tqdm

def inference():
    """
    inference on an ELECTRA discriminator model. should return embedding from discriminator

    This function sets up and executes the training process for an ELECTRA discriminator model, using a sequence of generator models provided by the user. It handles argument parsing, dataset loading, model initialization, and the training loop.

    The function performs the following steps:
    1. Parse command-line arguments for training configuration.
    2. Load and prepare the training and test datasets.
    3. Create data loaders for efficient batch processing.
    4. Initialize the ELECTRA generator and discriminator models.
    5. Set up the ELECTRA trainer with the specified parameters.
    6. Execute the training loop, which involves:
       - Loading different generator models (if specified)
       - Training the discriminator for a set number of epochs
       - Saving the model checkpoints

    The training process can be customized through various command-line arguments,
    including model architecture, training hyperparameters, and data paths.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--train_dataset", required=True, type=str, help="train dataset for train electra")
    parser.add_argument("-t", "--test_dataset", type=str, default=None, help="test set for evaluate train set")
    parser.add_argument("-v", "--vocab_path", required=True, type=str, help="built vocab model path with electra-vocab")
    parser.add_argument("-o", "--output_path", required=True, type=str, help="ex)output/")

    parser.add_argument("-hs", "--hidden", type=int, default=100, help="hidden size of transformer model")
    parser.add_argument("-l", "--layers", type=int, default=5, help="number of layers")
    parser.add_argument("-a", "--attn_heads", type=int, default=10, help="number of attention heads")
    parser.add_argument("-s", "--seq_len", type=int, default=1898, help="maximum sequence len")

    parser.add_argument("-ghs", "--gen_hidden", type=int, default=100, help="hidden size of gen transformer model")
    parser.add_argument("-gl", "--gen_layers", type=int, default=10, help="number of gen layers")
    parser.add_argument("-ga", "--gen_attn_heads", type=int, default=10, help="number of gen attention heads")
    parser.add_argument("-gs", "--gen_seq_len", type=int, default=1898, help="maximum gen sequence len")

    parser.add_argument("-b", "--batch_size", type=int, default=3, help="number of batch_size")
    parser.add_argument("-e", "--epochs", type=int, default=10,
                        help="number of epochs to train discriminator for each generator")
    parser.add_argument("-w", "--num_workers", type=int, default=0, help="dataloader worker size")
    parser.add_argument("--freeze", dest='freeze_embed', action='store_true',
                        help="freeze discriminator embedding layer to GloVE embeddings")
    parser.add_argument("--no_freeze", dest='freeze_embed', action='store_false',
                        help="train discriminator embedding layer after initializing to GloVE embeddings")
    parser.set_defaults(freeze_embed=False)

    parser.add_argument("--cuda", dest='with_cuda', action='store_true', help="train with CUDA")
    parser.add_argument("--no_cuda", dest='with_cuda', action='store_false', help="train on CPU")
    parser.set_defaults(with_cuda=False)

    parser.add_argument("--log_freq", type=int, default=100, help="printing loss every n iter: setting n")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")
    parser.add_argument("--d_log_file", type=str, default=None, help="log file for discriminator performance metrics")

    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate of adam")
    parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")

    parser.add_argument("--load_disc", type=str, nargs='+', default=None,
                        help="path to saved state_dicts of Masked LM model")
    parser.add_argument("--load_embed", type=str, nargs='+', default=None,
                        help="path to saved state_dicts for generator embeddings")
    parser.add_argument("--resume_epoch", type=int, default=0, help="epoch to resume training at")

    args = parser.parse_args()

    print("Loading Test Dataset", args.test_dataset)
    test_dataset = ELECTRADataset(args.test_dataset, args.vocab_path, original=True) if args.test_dataset is not None else None

    print("Creating Dataloader")
    test_data_loader = DataLoader(test_dataset, batch_size=64, num_workers=args.num_workers, drop_last=False) \
        if test_dataset is not None else None
    vocab_len = test_dataset.vocab_len()

    electra_config = ElectraConfig(vocab_size=vocab_len, embedding_size=args.hidden, hidden_size=2 * args.hidden,
                                   num_hidden_layers=args.layers, num_attention_heads=args.attn_heads,
                                   intermediate_size=4 * args.hidden, max_position_embeddings=args.seq_len)

    electra_disc = ElectraDiscriminator(electra_config, torch.from_numpy(test_dataset.embeddings), args.load_disc[0],
                                        args.load_embed[0])

    print(electra_disc)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    electra_disc.to(device)

    # Set the model to evaluation mode
    electra_disc.eval()

    # List to store predictions
    all_embeddings = []

    # Disable gradient computation for inference
    with torch.no_grad():
        # Iterate over the test DataLoader
        for batch in tqdm(test_data_loader):
            # create attention mask
            zero_boolean = torch.eq(batch["species_frequencies"], 0).to(device)
            mask = torch.ones(zero_boolean.shape, dtype=torch.float).to(device)
            mask = mask.masked_fill(zero_boolean, 0)

            data = batch['electra_input'].to(device) if isinstance(batch, dict) else batch[0].to(device)

            # If your model requires additional inputs (e.g., attention_mask), pass them as well:
            # attention_mask = batch['attention_mask'].to(device) if isinstance(batch, dict) else batch[1].to(device)
            # outputs = electra_disc(input_ids, attention_mask=attention_mask)

            # Here, we assume electra_disc returns logits.
            outputs = electra_disc.forward_embeddings(data, mask)
            all_embeddings.append(outputs['hidden_states'][-1][:, 0, :].detach().cpu().numpy())
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    # save all embeddings to somewhere
    save_path = "/home/kevin/Desktop/gut_microbiome/outputs_hmc/random_forest_pred" # hard coded cuz lazy
    np.save(save_path, all_embeddings)


if __name__ == "__main__":
    inference()
