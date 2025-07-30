import os
import argparse
import numpy as np
import anndata as ad
import pandas as pd
from skbio.stats.composition import clr, closure, multi_replace

def compute_prevalence_abundance_adata(adata: ad.AnnData):
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()

    prevalence = np.mean(X > 0, axis=0)
    abundance = np.mean(X, axis=0)

    return (
        pd.Series(prevalence, index=adata.var_names),
        pd.Series(abundance, index=adata.var_names),
    )

def preprocess_clr_matrix(X):
    if hasattr(X, "toarray"):
        X = X.toarray()
    X_replaced = multi_replace(X)
    X_closed = closure(X_replaced)
    return clr(X_closed)

def main():
    parser = argparse.ArgumentParser(description="Filter and CLR-transform a taxa abundance matrix using scikit-bio.")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--train-input", type=str, required=True)
    parser.add_argument("--test-input", type=str, required=True)
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
    prevalence, abundance = compute_prevalence_abundance_adata(adata_combined)

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
    X_dense = preprocess_clr_matrix(X_dense)
    
    # Split using metadata tag
    adata_combined.obsm["embedding"] = X_dense

    train_filtered = adata_combined[adata_combined.obs["__split"] == "train"].copy()
    test_filtered = adata_combined[adata_combined.obs["__split"] == "test"].copy()
    
    for ds in [train_filtered, test_filtered]:
        del ds.obs["__split"]

    os.makedirs(args.output_dir, exist_ok=True)

    # Save filtered objects with CLR in layers
    train_path = os.path.join(args.output_dir, "raw_encoded_train.h5ad")
    test_path = os.path.join(args.output_dir, "raw_encoded_test.h5ad")

    train_filtered.write_h5ad(train_path)
    test_filtered.write_h5ad(test_path)

    print(f"Saved CLR-transformed train data to {train_path}, shape: {train_filtered.obsm['embedding'].shape}")
    print(f"Saved CLR-transformed test data to {test_path}, shape: {test_filtered.obsm['embedding'].shape}")

if __name__ == "__main__":
    main()
