import numpy as np
import pandas as pd
import os
# import fasttext
import argparse
from sklearn.model_selection import train_test_split
from collections import defaultdict
import anndata as ad
import scipy.sparse as sp

# os.environ["GOOGLE_API_KEY"] = "AIzaSyB41iEts_InBYR3sHz1bywFYN2JjxlBTJ0"
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# if not GOOGLE_API_KEY:
#     raise ValueError("Please set your GOOGLE_API_KEY environment variable.")


def get_loc_labels(samples_list, sample_metadata_path):
    """
    inputs:
    a list of sample names,
    a path to sample_metadata.tsv, a file containing location info of all samples,
    return a list of locations corresponding to the samples
    """
    metadata_df = pd.read_csv(sample_metadata_path, sep='\t')
    metadata_df["sample_id"] = metadata_df["project"] + "_" + metadata_df["srr"]
    metadata_df = metadata_df.set_index("sample_id")
    location_list = []
    for sample_id in samples_list:
        row = metadata_df.loc[sample_id]
        location_list.append(row.region)
    return location_list


def add_top_k_layer(adata, k=512):
    """
    Adds a new data layer to the AnnData object containing only the top-k features (taxa) per sample.
    Assumes `adata.X` is dense or sparse with shape (n_samples, n_taxa).

    Parameters:
        adata: AnnData object
        k: number of top features to keep per sample
        layer_name: name of the layer to store the result in
    """
    X = adata.X.toarray() if sp.issparse(adata.X) else adata.X
    n_samples = X.shape[0]

    # Create a new matrix with only top-k taxa per sample
    top_k_matrix = np.zeros_like(X)

    for i in range(n_samples):
        row = X[i]
        topk_indices = np.argsort(-row)[:k]  # Indices of top-k taxa
        top_k_matrix[i, topk_indices] = row[topk_indices]

    # Optionally make it sparse
    adata.layers[f"top_{k}"] = sp.csr_matrix(top_k_matrix)

    # Stats
    original_nonzero = (X > 0).sum()
    retained_nonzero = (top_k_matrix > 0).sum()
    print(f"Top {k} layer retained {retained_nonzero / original_nonzero:.2%} of original non-zero entries")


def save_taxonomy_table(adata, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    adata.write(save_path)
    print(f"Saved to {save_path} and numpy array with shape {adata.shape}")


def train_test_split_anndata(adata, test_size=0.2, random_state=42, stratify_obskey=None):
    """
    Splits an AnnData object into train and test sets.

    Parameters
    ----------
    adata : AnnData
        The AnnData object to split.
    test_size : float, optional
        Proportion of the dataset to include in the test split.
    random_state : int, optional
        Seed used by the random number generator.
    stratify_obskey : str or None, optional
        If not None, use this key from adata.obs for stratified sampling.

    Returns
    -------
    adata_train : AnnData
        Training subset of the data.
    adata_test : AnnData
        Test subset of the data.
    """
    if stratify_obskey is not None:
        if stratify_obskey not in adata.obs.columns:
            raise ValueError(f"{stratify_obskey} not found in adata.obs columns.")
        stratify_labels = adata.obs[stratify_obskey]
    else:
        stratify_labels = None

    train_idx, test_idx = train_test_split(
        adata.obs_names,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_labels
    )

    adata_train = adata[train_idx].copy()
    adata_test = adata[test_idx].copy()

    return adata_train, adata_test


def split_anndata_by_study(adata, remove_agp=False, split_ratio=0.5):
    """
    Splits the AnnData object into two groups, each with roughly {split_ratio} of the total samples.
    All samples from the same study (adata.obs['study_id']) are kept in the same group.

    Parameters:
      - adata: AnnData object with 'study_id' in .obs
      - remove_agp: Whether to exclude study "PRJEB11419"
      - split_ratio: Approximate fraction of samples to assign to group 1

    Returns:
      - adata_group1: AnnData object for group 1
      - adata_group2: AnnData object for group 2
    """
    # Get sample counts per study
    study_counts = adata.obs["study_id"].value_counts().to_dict()

    # Optionally remove AGP
    if remove_agp and "PRJEB11419" in study_counts:
        print(f"Removing {study_counts['PRJEB11419']} samples from American Gut Project (PRJEB11419).")
        del study_counts["PRJEB11419"]

    # Sort studies by count descending
    sorted_studies = sorted(study_counts.items(), key=lambda x: x[1], reverse=True)

    # Greedily assign studies to balance group sizes
    group1_studies, group2_studies = [], []
    count1, count2 = 0, 0

    for study, count in sorted_studies:
        if count1 < (count1 + count2) * split_ratio:
            group1_studies.append(study)
            count1 += count
        else:
            group2_studies.append(study)
            count2 += count

    # Boolean masks for group selection
    mask_group1 = adata.obs["study_id"].isin(group1_studies)
    mask_group2 = adata.obs["study_id"].isin(group2_studies)

    # Create new AnnData objects
    adata_group1 = adata[mask_group1].copy()
    adata_group2 = adata[mask_group2].copy()

    return adata_group1, adata_group2


def read_taxonomic_table(file_path, nrows = None):
    """takes a file path of a csv file containing the hmc taxonomic table
    returns:
    - a numpy array of data of shape (n_rows, n_cols, 2) where [:, :, 0] represents the index of column and [:, :, 1]
      represents actual table value
    - a list of column names (bacteria names)
    - a list of row names (sample names)
    """
    df = pd.read_csv(file_path, nrows=nrows)
    df = df.drop(columns=["Unnamed: 0"])
    df = df.set_index("sample")
    # 1. shuffle the dataframe
    df = df.sample(frac=1, random_state=42)
    adata = ad.AnnData(X=df)
    adata.obs["sample"] = adata.obs_names
    adata.var["taxa"] = adata.var_names
    # Extract study_id from sample_name.
    adata.obs["study_id"] = adata.obs["sample"].apply(lambda x: x.split('_')[0])
    add_top_k_layer(adata, k=512)  # Add top 512 layer
    return adata


def main():
    parser = argparse.ArgumentParser(description="Preprocess HMC taxonomic table.")
    parser.add_argument('--taxonomic_table_path', type=str, required=True, help='Path to the taxonomic table CSV file.')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save pretrain data.')
    parser.add_argument('--anndata_pretrain_filename', type=str, default="taxonomy_table_pretrain", help='Pretrain .h5ad file name.')
    parser.add_argument('--anndata_finetune_filename', type=str, default="taxonomy_table_finetune", help='Finetune .h5ad file name.')
    parser.add_argument('--sample_metadata_path', type=str, required=True, help='Finetune .h5ad file name.')

    parser.add_argument('--split_ratio', type=float, default=0.8, help='Ratio for splitting the dataset into pretrain and finetune sets.')
    parser.add_argument('--nrows', type=int, default=None, help='Number of rows to read from the taxonomic table CSV file.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility.')
    parser.add_argument('--split_finetune_by_study', action='store_true', help='Whether to split finetune data by study.')
    parser.add_argument('--add_loc_labels', action='store_true', help='Whether to add location labels to the data.')
    args = parser.parse_args()

    taxonomic_table_path = args.taxonomic_table_path
    save_dir = args.save_dir
    anndata_pretrain_filename = args.anndata_pretrain_filename
    anndata_finetune_filename = args.anndata_finetune_filename
    sample_metadata_path = args.sample_metadata_path
    nrows = args.nrows
    split_ratio = args.split_ratio
    split_finetune_by_study = args.split_finetune_by_study
    add_loc_labels = args.add_loc_labels
    
    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"Setting random seed to {args.seed} for reproducibility.")

    print(f"Reading taxonomic table from {taxonomic_table_path} with nrows={nrows} and split_ratio={split_ratio}")
    adata = read_taxonomic_table(taxonomic_table_path, nrows=nrows)
    
    if add_loc_labels:
        print("Adding location labels to pretrain and finetune data.")
        adata.obs["location"] = get_loc_labels(adata.obs["sample"], sample_metadata_path)
    
    pretrain_adata, finetune_adata = split_anndata_by_study(adata, remove_agp=True, split_ratio=split_ratio)

    save_taxonomy_table(pretrain_adata, os.path.join(save_dir, anndata_pretrain_filename))
    save_taxonomy_table(finetune_adata, os.path.join(save_dir, anndata_finetune_filename))

    # train test split for finetune data
    if split_finetune_by_study:
        train_adata, test_adata = split_anndata_by_study(finetune_adata, split_ratio=0.8)
    else:
        train_adata, test_adata = train_test_split_anndata(finetune_adata, test_size=0.2, random_state=42)

    save_taxonomy_table(train_adata, os.path.join(save_dir, "finetune_data_train.h5ad"))
    save_taxonomy_table(test_adata, os.path.join(save_dir, "finetune_data_test.h5ad"))


if __name__ == '__main__':
    main()
    # # train test split for finetune data
    # finetune_adata = ad.read_h5ad("/project/aip-rahulgk/gutmodel/datasets_halfsplit/taxonomy_table_finetune.h5ad")
    # save_dir = "/project/aip-rahulgk/gutmodel/datasets_halfsplit/nonstudy_split"
    # train_adata, test_adata = train_test_split_anndata(finetune_adata, test_size=0.2, random_state=42, stratify_obskey="location")

    # save_taxonomy_table(train_adata, os.path.join(save_dir, "finetune_data_train.h5ad"))
    # save_taxonomy_table(test_adata, os.path.join(save_dir, "finetune_data_test.h5ad"))