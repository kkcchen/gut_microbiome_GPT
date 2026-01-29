import argparse
import anndata as ad

from data_utils.preprocessor import preprocess_clr_matrix, compute_prevalence_abundance
import torch

def clr_torch(X, pseudocount: float = 1e-6):

        x = X.to(torch.float32)
        out = torch.zeros_like(x)
        
        # Add pseudocount
        xv = x + pseudocount
        logx = torch.log(xv.clamp_min(pseudocount))  # (B, L)
        
        # denom = xv.sum(dim=1, keepdim=True).clamp_min(pseudocount)
        # out = logx - torch.log(denom)

        # Per-sample mean of log counts
        gm = logx.mean(dim=1, keepdim=True)

        # Fill only valid positions; invalid remain 
        out = (logx - gm)

        # keep pad/class positions at 0.0
        return out

def main():
    parser = argparse.ArgumentParser(description="Filter and CLR-transform a taxa abundance matrix using scikit-bio.")
    parser.add_argument("--train-input", type=str, required=True)
    parser.add_argument("--test-input", type=str, required=True)
    parser.add_argument("--train-output", type=str, required=True)
    parser.add_argument("--test-output", type=str, required=True)
    parser.add_argument("--prevalence-threshold", type=float, default=0.01)
    parser.add_argument("--abundance-threshold", type=float, default=0.05)
    args = parser.parse_args()

    # Load AnnData
    adata_train = ad.read_h5ad(args.train_input)
    adata_test = ad.read_h5ad(args.test_input)
    
    adata_train.obs["__split"] = "train"
    adata_test.obs["__split"] = "test"

    # Concatenate train and test data
    adata_combined = ad.concat([adata_train, adata_test], axis=0)
    adata_combined.var["taxa"] = adata_combined.var_names
    # Compute prevalence and abundance
    prevalence, abundance = compute_prevalence_abundance(adata_combined)

    # Filter taxa
    keep_mask = (prevalence >= args.prevalence_threshold) & (abundance >= args.abundance_threshold)
    kept_taxa = keep_mask[keep_mask].index.tolist()

    if len(kept_taxa) == 0:
        raise ValueError("No taxa passed the filtering thresholds.")

    X_dense = adata_combined[:, kept_taxa].X
    nonzero_sample_mask = (X_dense > 0).any(axis=1)
    adata_combined = adata_combined[nonzero_sample_mask].copy()
    if adata_combined.n_obs == 0:
        raise ValueError("All samples became zero after filtering. Check your thresholds.")
    
    # Re-extract X after filtering both taxa and samples
    X_dense = adata_combined[:, kept_taxa].X
    # Compute CLR
########################
    # X_dense = preprocess_clr_matrix(X_dense)
    X_dense  = clr_torch(torch.from_numpy(X_dense.toarray() if not isinstance(X_dense,  torch.Tensor) else X_dense)).numpy()
##################    
    # Split using metadata tag
    adata_combined.obsm["raw_embedding"] = X_dense

    train_filtered = adata_combined[adata_combined.obs["__split"] == "train"].copy()
    test_filtered = adata_combined[adata_combined.obs["__split"] == "test"].copy()
    
    for ds in [train_filtered, test_filtered]:
        del ds.obs["__split"]

    # Save filtered objects with CLR in layers
    train_filtered.write_h5ad(args.train_output)
    test_filtered.write_h5ad(args.test_output)

    print(f"Saved CLR-transformed train data to {args.train_input}, shape: {train_filtered.obsm['raw_embedding'].shape}")
    print(f"Saved CLR-transformed test data to {args.test_input}, shape: {test_filtered.obsm['raw_embedding'].shape}")

if __name__ == "__main__":
    main()
