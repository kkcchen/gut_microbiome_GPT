import numpy as np
import pandas as pd
import os
# import fasttext
import argparse
from sklearn.model_selection import train_test_split
from collections import defaultdict
import anndata as ad
import scipy.sparse as sp
from skbio.stats.composition import clr, closure, multi_replace

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

def label_sex_from_tags(
    adata: ad.AnnData,
    tags_df,
    studies,
    tag_names,
    male_indicators,
    female_indicators,
):
    assert len(studies) == len(tag_names) == len(male_indicators) == len(female_indicators), \
        "All input lists must have the same length"
        
    obs = adata.obs.copy()
    indices_to_keep = []
    obs["label"] = pd.NA
    for study, tag_name, male_val, female_val in zip(studies, tag_names, male_indicators, female_indicators):
        filtered_tags = tags_df[(tags_df["project"] == study) & (tags_df["tag"] == tag_name)]
        sample_to_value = dict(zip(filtered_tags["srr"], filtered_tags["value"]))
        mask = adata.obs["drr"].isin(sample_to_value.keys())
        # Assign male/female based on value
        obs.loc[mask & (obs["drr"].map(sample_to_value) == male_val), "label"] = "male"
        obs.loc[mask & (obs["drr"].map(sample_to_value) == female_val), "label"] = "female"

        indices_to_keep.extend(obs.index[mask].tolist())

    indices_to_keep = list(dict.fromkeys(indices_to_keep))
    adata_subset = adata[indices_to_keep].copy()
    adata_subset.obs["label"] = obs.loc[indices_to_keep, "label"].fillna("other")
    
    # Print total "other" count
    other_count = (adata_subset.obs["label"] == "other").sum()
    print(f"Total 'other' labels: {other_count}")
    return adata_subset

def split_studies_and_tags(adata: ad.AnnData, tags_df, studies: list, tag_names: list, force_numeric=False):
    assert len(studies) == len(tag_names), "studies and tag_names must be same length"
    indices_to_keep = []
    obs = adata.obs.copy()
    obs['label'] = pd.NA
    for study, tag_name in zip(studies, tag_names):
        filtered_tags = tags_df[(tags_df['project'] == study) & (tags_df['tag'] == tag_name)]
        sample_to_value = dict(zip(filtered_tags['srr'], filtered_tags['value']))
        mask = adata.obs["drr"].isin(sample_to_value.keys())
        obs.loc[mask, 'label'] = adata.obs["drr"][mask].map(sample_to_value)
        indices_to_keep.extend(obs.index[mask].tolist())
    indices_to_keep = list(dict.fromkeys(indices_to_keep))
    adata_subset = adata[indices_to_keep].copy()
    adata_subset.obs['label'] = obs.loc[indices_to_keep, 'label']
    
    if force_numeric:
        adata_subset.obs['label'] = pd.to_numeric(adata_subset.obs['label'], errors='coerce')
        coerced_count = adata_subset.obs['label'].isna().sum()
        print(f"Number of coerced values: {coerced_count}")
        adata_subset = adata_subset[adata_subset.obs['label'].notna()].copy()
        
    return adata_subset


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


def save_adata(adata, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    adata.write(save_path)
    print(f"Saved to {save_path} and numpy array with shape {adata.shape}")
    
def restore_adata(save_path):
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"File {save_path} does not exist.")
    
    adata = ad.read_h5ad(save_path)
    print(f"Restored AnnData object with shape {adata.shape} from {save_path}")
    return adata


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

def split_anndata_specify_study(adata: ad.AnnData, studies_to_isolate: list):
    mask = adata.obs['study_id'].isin(studies_to_isolate)
    adata_in = adata[mask].copy()
    adata_out = adata[~mask].copy()
    return adata_in, adata_out


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
    adata.obs["drr"] = [idx.split('_')[1] for idx in adata.obs.index]
    adata.var["taxa"] = adata.var_names
    adata.obs["study_id"] = [idx.split('_')[0] for idx in adata.obs.index]
    add_top_k_layer(adata, k=512)  # Add top 512 layer
    return adata


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
    parser = argparse.ArgumentParser(description="Preprocess HMC taxonomic table.")
    parser.add_argument('--taxonomic_table_path', type=str, required=True, help='Path to the taxonomic table CSV file.')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save pretrain data.')
    parser.add_argument('--sample_metadata_path', type=str, required=True, help='metadata path')
    parser.add_argument('--tags_path', type=str, default="data/tags.tsv", help='Path to the tags file.')

    parser.add_argument('--split_ratio', type=float, default=0.8, help='Ratio for splitting the dataset into pretrain and finetune sets.')
    parser.add_argument('--nrows', type=int, default=None, help='Number of rows to read from the taxonomic table CSV file.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility.')
    parser.add_argument('--split_finetune_by_study', action='store_true', help='Whether to split finetune data by study.')
    parser.add_argument('--add_loc_labels', action='store_true', help='Whether to add location labels to the data.')
    parser.add_argument("--prevalence-threshold", type=float, default=0.01)
    parser.add_argument("--abundance-threshold", type=float, default=0.05)
    args = parser.parse_args()

    taxonomic_table_path = args.taxonomic_table_path
    save_dir = args.save_dir
    sample_metadata_path = args.sample_metadata_path
    tags_path = args.tags_path
    nrows = args.nrows
    split_ratio = args.split_ratio
    split_finetune_by_study = args.split_finetune_by_study
    add_loc_labels = args.add_loc_labels
    
    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"Setting random seed to {args.seed} for reproducibility.")

    print(f"Reading taxonomic table from {taxonomic_table_path} with nrows={nrows} and split_ratio={split_ratio}")
    adata = read_taxonomic_table(taxonomic_table_path, nrows=nrows)
    
    prevalence, abundance = compute_prevalence_abundance_adata(adata)
    keep_mask = (prevalence >= args.prevalence_threshold) & (abundance >= args.abundance_threshold)
    kept_taxa = keep_mask[keep_mask].index.tolist()
    if len(kept_taxa) == 0:
        raise ValueError("No taxa passed the filtering thresholds.")

    # raw filtering
    X_dense = adata[:, kept_taxa].X
    nonzero_sample_mask = (X_dense > 0).any(axis=1)
    adata = adata[nonzero_sample_mask].copy()
    if adata.n_obs == 0:
        raise ValueError("All samples became zero after filtering. Check your thresholds.")
    X_dense = adata[:, kept_taxa].X
    X_dense = preprocess_clr_matrix(X_dense)
    adata.obsm["raw_embedding"] = X_dense
    
    if add_loc_labels:
        print("Adding location labels to pretrain and finetune data.")
        adata.obs["location"] = get_loc_labels(adata.obs.index, sample_metadata_path)
        
    age_studies = ['PRJNA729511', 'PRJEB5729', 'PRJNA485316']
    age_cols = ['age', 'age', 'host_age']
    
    sex_studies = ['PRJNA729511', 'PRJNA516932', 'PRJNA559143', 'PRJEB5482']
    sex_cols = ['sex', 'host_sex', 'gender_cat', 'sex']
    male_indicators = ['male', 'male', '1', 'male']
    female_indicators = ['female', 'female', '2', 'female']
    
    bmi_studies = ['PRJNA485316', 'PRJNA559143', 'PRJNA516932', 'PRJEB6702', 'PRJEB5729']
    bmi_cols = ['host_body_mass_index', 'body_mass_index', 'host_body_mass_index', 'body_mass_index', 'body_mass_index']
    
    supplement_studies = ['PRJNA428736']
    supplement_cols = ['supplement']
    
    mice_studies = ['PRJEB4244']
    mice_cols = ['diet']
    
    diet_studies = ['PRJEB5729']
    diet_cols = ['diet']
    
    AGP_study = 'PRJEB11419'
    
    all_studies = age_studies + sex_studies + bmi_studies + supplement_studies + mice_studies + diet_studies + [AGP_study]
    
    finetune_adata, pretrain_adata = split_anndata_specify_study(adata, all_studies)
    # pretrain_adata, finetune_adata = split_anndata_by_study(adata, remove_agp=True, split_ratio=split_ratio)

    # pretrain_adata = restore_adata(os.path.join(save_dir, anndata_pretrain_filename))
    # finetune_adata = restore_adata(os.path.join(save_dir, anndata_finetune_filename))
    
    labeled_datasets = {}
    tags_df = pd.read_csv(tags_path, sep='\t')

    # Age
    labeled_datasets['age'] = {
        "adata": split_studies_and_tags(
            finetune_adata, tags_df,
            studies=age_studies,
            tag_names=age_cols,
            force_numeric=True
        ),
        "stratify": False
    }

    # BMI
    labeled_datasets['bmi'] = {
        "adata": split_studies_and_tags(
            finetune_adata, tags_df,
            studies=bmi_studies,
            tag_names=bmi_cols,
            force_numeric=True
        ),
        "stratify": False
    }

    # Supplement
    labeled_datasets['supplement'] = {
        "adata": split_studies_and_tags(
            finetune_adata, tags_df,
            studies=supplement_studies,
            tag_names=supplement_cols
        ),
        "stratify": True
    }

    # Mice diet
    labeled_datasets['mice_diet'] = {
        "adata": split_studies_and_tags(
            finetune_adata, tags_df,
            studies=mice_studies,
            tag_names=mice_cols
        ),
        "stratify": True
    }

    # Human diet
    labeled_datasets['human_diet'] = {
        "adata": split_studies_and_tags(
            finetune_adata, tags_df,
            studies=diet_studies,
            tag_names=diet_cols
        ),
        "stratify": True
    }
    
    # Combine omnivores into one class, others into "non_omnivore"
    omnivore_like = {
        "omnivore",
        "omnivore.no.red.meat",
        "omnivore.but.no.red.meat"
    }
    labeled_datasets['human_diet']['adata'].obs['label'] = labeled_datasets['human_diet']['adata'].obs['label'].apply(
        lambda x: "omnivore" if x in omnivore_like else "non_omnivore"
    )

    # Sex
    labeled_datasets['sex'] = {
        "adata": label_sex_from_tags(
            finetune_adata, tags_df,
            studies=sex_studies,
            tag_names=sex_cols,
            male_indicators=male_indicators,
            female_indicators=female_indicators
        ),
        "stratify": True
    }

    # train test split for location
    train_adata, location_adata = split_anndata_by_study(pretrain_adata, split_ratio=0.7)
    ad.concat([location_adata, finetune_adata])
    location_train_adata, location_test_adata = split_anndata_by_study(location_adata, split_ratio=0.8)
    
    save_adata(train_adata, os.path.join(save_dir, "pretrain.h5ad"))
    save_adata(location_train_adata, os.path.join(save_dir, "finetune_loc_train.h5ad"))
    save_adata(location_test_adata, os.path.join(save_dir, "finetune_loc_test.h5ad"))
    
    
    print(location_train_adata.obs['location'].value_counts())
    print(location_test_adata.obs['location'].value_counts())
    
    all_train = []
    all_test = []
    
    for key, dataset_dict in labeled_datasets.items():
        dataset = dataset_dict["adata"]
        print(f"Processing dataset for {key}")
        print(dataset)
        print(f"for the label, unique values counts are: {dataset.obs['label'].value_counts()}")
        
        label_dtype = dataset.obs["label"].dtype

        if label_dtype.kind in {"f", "i"}:  # float or int → continuous label
            dataset.obs = dataset.obs.rename(columns={"label": "continuous_label"})
            dataset.obs["categorical_label"] = pd.NA
            stratify_key = "continuous_label" if dataset_dict.get("stratify", False) else None
        else:
            dataset.obs["categorical_label"] = dataset.obs["label"].astype(str)
            dataset.obs = dataset.obs.drop(columns=["label"])
            dataset.obs["continuous_label"] = np.nan
            stratify_key = "categorical_label" if dataset_dict.get("stratify", False) else None
        
        train_adata, test_adata = train_test_split_anndata(
            dataset,
            test_size=0.2,
            random_state=42,
            stratify_obskey=stratify_key
        )
        train_adata.obs["downstream_task"] = key
        test_adata.obs["downstream_task"] = key
        
        all_train.append(train_adata)
        all_test.append(test_adata)
    
    train_all = ad.concat(all_train, join="outer")
    test_all = ad.concat(all_test, join="outer")
    
    print(f"Final train data: {train_all}")
    print(f"Final test data: {test_all}")
    
    save_adata(train_all, os.path.join(save_dir, "finetune_train.h5ad"))
    save_adata(test_all, os.path.join(save_dir, "finetune_test.h5ad"))
    
    print(train_all.obs['downstream_task'].value_counts())
    print(test_all.obs['downstream_task'].value_counts())


if __name__ == '__main__':
    main()
    # # train test split for finetune data
    # finetune_adata = ad.read_h5ad("/project/aip-rahulgk/gutmodel/datasets_halfsplit/taxonomy_table_finetune.h5ad")
    # save_dir = "/project/aip-rahulgk/gutmodel/datasets_halfsplit/nonstudy_split"
    # train_adata, test_adata = train_test_split_anndata(finetune_adata, test_size=0.2, random_state=42, stratify_obskey="location")

    # save_adata(train_adata, os.path.join(save_dir, "finetune_data_train.h5ad"))
    # save_adata(test_adata, os.path.join(save_dir, "finetune_data_test.h5ad"))