import torch
import argparse
import os
import numpy as np
import json

from safetensors.torch import load_file
from data_utils.dataloader import prepare_dataloader

from accelerate import Accelerator
from trainers import logger
from trainers.test_functions import (
    restore_vocab_test,
    create_testdata_state,
)

from trainers.finetune_functions import (
    load_finetuned_model
)


def main():
    parser = argparse.ArgumentParser(description="Encode data with a model using safetensors weights.")
    parser.add_argument("--safetensors-path", type=str, required=True, help="Path to the .safetensors file")
    parser.add_argument("--output-path", type=str, required=True, help="Where to save the output tensor")
    parser.add_argument("--vocab-dir", type=str, required=True, help="dir of vocab h5ad")
    parser.add_argument("--model-config-path", type=str, required=True, help="Path to the model configuration file (not used in this script but can be useful for reference)")
    parser.add_argument("--adata-path", type=str, required=True, help="Path to the training data h5ad file")
    parser.add_argument("--is-finetune", action="store_true", help="Path to the training data numpy file")
    parser.add_argument("--nrows", type=int, default=None, help="Number of rows to use from the anndata file (for debugging)")
    parser.add_argument("--emb-colname", type=str, default="embedding", help="Column name for the embeddings in the output anndata file")
    args = parser.parse_args()
    
    safetensors_path = args.safetensors_path
    output_path = args.output_path
    vocab_dir = args.vocab_dir
    adata_path = args.adata_path
    model_config_path = args.model_config_path
    
    nrows = args.nrows
    
    # Initialize accelerator
    accelerator = Accelerator()

    # === Load model ===
    from models import TransformerModel
    
    if args.is_finetune:
        finetuned_model, model_config = load_finetuned_model(model_config_path, safetensors_path)
        model = finetuned_model.base_model
        num_bins = model.n_input_bins
        use_gnn = model_config["base_model_config"].get("use_gnn", False)
    else:
        with open(model_config_path, 'r') as f:
            model_config = json.load(f)
        model = TransformerModel(**model_config)
        state_dict = load_file(safetensors_path)
        model.load_state_dict(state_dict)
        use_gnn = model_config.get("use_gnn", False)
    model.eval()
    num_bins = model.n_input_bins
    
    # restore vocab
    vocab, _, adata, graph_data = restore_vocab_test(adata_path, vocab_dir, use_gnn, downstream_task=None, batch_obskey=None, nrows=nrows, accelerator=accelerator)

    # === Load test dataloader ===
    data_bin_strategy = model.bin_strategy  # Ensure consistency between model and data binning strategy
    data_dict, adata = create_testdata_state(
        adata=adata,
        num_bins=num_bins,
        vocab=vocab,
        batch_obskey=None,  # No batch key needed for encoding
        continuous_obskey=None,  # No continuous labels needed for encoding
        nrows=nrows,  # Set to None to use all rows
        bin_strategy=data_bin_strategy
    )
    
    logger.info(f"num obs in adata after filtering: {adata.n_obs}")
    
    dataloader = prepare_dataloader(
        data_dict,
        use_batch_labels=False,
        use_continuous_labels=False,
        gen_percent=0.0,  # No generation for encoding
        vocab=vocab,
        batch_size=64,  # Adjust batch size as needed
        shuffle=False,  # Do not shuffle for encoding
        do_subsample = False,
        do_clr = True, #### this is for the contrastive version
    )

    # Prepare model and dataloader with accelerate
    model, dataloader = accelerator.prepare(model, dataloader)

    # === Encode ===
    all_cell_embs = []
############
    all_obs_idx = []
############
    with torch.no_grad():
        for data_dict in dataloader:
            taxa = data_dict["ids"]
            values = data_dict["values"]
#############
            obs_idx = data_dict["obs_idx"]           # NEW
#############
            src_padding_mask = taxa.eq(vocab.pad_index)
            unwrapped_model = accelerator.unwrap_model(model)
            output, _ = unwrapped_model.encode(
                src=taxa,  # (batch, seq_len)
                values=values,  # (batch, seq_len)
                src_key_padding_mask=src_padding_mask,  # (batch, seq_len)
                graph_data=graph_data,  # Graph data if applicable
            )  # (batch, seq_len, embsize)
            cell_emb = unwrapped_model.get_cell_emb_from_layer(output)  # (batch, embsize)
            # gathered = accelerator.gather_for_metrics(cell_emb)  # Gather across processes
################
            obs_idx_g, gathered = accelerator.gather_for_metrics((obs_idx, cell_emb))
            all_obs_idx.append(obs_idx_g.cpu())
################
            all_cell_embs.append(gathered.cpu())

    # === Save result (only main process) ===
    if accelerator.is_main_process:
        # final_tensor = torch.cat(all_cell_embs, dim=0)
        # adata.obsm[args.emb_colname] = np.array(final_tensor)
########################
        obs_idx_cat = torch.cat(all_obs_idx, dim=0).numpy()
        emb_cat = torch.cat(all_cell_embs, dim=0).numpy()

        # Sort by obs_idx to guarantee alignment
        order = np.argsort(obs_idx_cat)
        obs_idx_sorted = obs_idx_cat[order]
        emb_sorted = emb_cat[order]

        # PROOF CHECK: obs_idx should be exactly 0..n_obs-1
        assert np.array_equal(obs_idx_sorted, np.arange(adata.n_obs)), \
            "Embedding rows do not match adata row order!"

        adata.obsm[args.emb_colname] = emb_sorted
#########################
        print("after:", adata)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        adata.write_h5ad(output_path)
        print(f"Saved cell embeddings as anndata to {output_path} with column {args.emb_colname}")


if __name__ == "__main__":
    main()
