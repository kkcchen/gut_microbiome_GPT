import os
import torch
import numpy as np
import anndata as ad
from data_utils.vocab import MicrobiomeVocab, BatchVocab
from data_utils.preprocessor import Preprocessor
from data_utils.graph_helpers import build_tg_data_from_taxon_df
from trainers import logger

def prepare_data_artifacts(
    ann_table_path, 
    data_restore_dir, 
    accelerator, 
    use_gnn=False, 
    batch_obskey=None, 
    nrows=None,
    preprocess_strategy="default", # "default" (scPRINT/Continuous) or "binning"
    num_bins=15
):
    """
    Ensures that processed data artifacts (vocab, split h5ad, graph) exist on disk.
    Returns the paths to these artifacts so the Dataset can load them lazily.
    """
    
    # Define paths
    vocab_path = os.path.join(data_restore_dir, "vocab_file.json")
    batchvocab_path = os.path.join(data_restore_dir, f"batchvocab_{batch_obskey}.json")
    processed_h5ad_path = os.path.join(data_restore_dir, "augmented_data.h5ad")
    graph_path = os.path.join(data_restore_dir, "graph_data.pt")

    # Only Main Process performs the IO and Preprocessing
    if accelerator.is_main_process:
        os.makedirs(data_restore_dir, exist_ok=True)
        
        # Check if already processed
        artifacts_exist = os.path.exists(vocab_path) and os.path.exists(processed_h5ad_path)
        if batch_obskey: artifacts_exist &= os.path.exists(batchvocab_path)
        if use_gnn: artifacts_exist &= os.path.exists(graph_path)

        if artifacts_exist:
            logger.info(f"Restoring data state from {data_restore_dir}")
            check_ad = ad.read_h5ad(processed_h5ad_path, backed='r')
            assert "split" in check_ad.obs, "Restored AnnData missing 'split' column."
        else:
            logger.info(f"Creating data state from scratch using {ann_table_path}")
            
            # 1. Load Raw
            adata = ad.read_h5ad(ann_table_path)
            if nrows: adata = adata[:nrows, :].copy()

            # 2. Vocab
            vocab = MicrobiomeVocab.create_vocab_from_scratch(adata)
            vocab.save_vocab(vocab_path)

            # 3. Batch Vocab
            if batch_obskey:
                batch_vocab = BatchVocab.create_batchvocab_from_scratch(batch_obskey, adata)
                batch_vocab.save_batchvocab(batchvocab_path)
                adata = batch_vocab.assign_batchvocab(adata)

            # 4. Preprocessing Strategy
            preprocessor = Preprocessor(binning=num_bins)
            hmc_npy = np.array(adata.X, dtype=np.float32)
            taxa_ids = np.array(adata.var["taxa_id"])

            if preprocess_strategy == "binning":
                # Legacy: Discretize values 0-14
                processed_values, _ = preprocessor.bin_from_np(hmc_npy, taxa_ids)
                adata.layers["processed_values"] = processed_values # Save as INTs
            
            elif preprocess_strategy == "default":
                # scPRINT style: CLR Normalization (Continuous)
                # Note: We do NOT filter to top 512 here. That happens in the Collator/Loader.
                # We want to save the "clean" full continuous data.
                processed_values = preprocessor.clr_from_np(hmc_npy, taxa_ids)
                adata.layers["processed_values"] = processed_values # Save as FLOATs
            
            else:
                raise ValueError(f"Unknown strategy: {preprocess_strategy}")

            # 5. Graph
            if use_gnn:
                graph_data = build_tg_data_from_taxon_df(adata.varm['taxonomy'], vocab.vocab_list)
                torch.save(graph_data, graph_path)

            # 6. Create Splits (Train/Val)
            n_obs = adata.n_obs
            train_indices = np.random.choice(n_obs, size=int(n_obs * 0.8), replace=False)
            is_train = np.zeros(n_obs, dtype=bool)
            is_train[train_indices] = True            
            adata.obs["split"] = np.where(is_train, "train", "val")

            # 7. Save to Disk
            adata.write_h5ad(processed_h5ad_path)
            logger.info(f"Saved processed artifacts to {data_restore_dir}")

    # Barrier: Wait for main process to finish writing files
    accelerator.wait_for_everyone()

    return {
        "h5ad_path": processed_h5ad_path,
        "vocab_path": vocab_path,
        "graph_path": graph_path if use_gnn else None,
        "batchvocab_path": batchvocab_path if batch_obskey else None
    }