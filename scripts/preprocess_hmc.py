import numpy as np
import pandas as pd
import os
# import fasttext
import json
import argparse
from sklearn.model_selection import train_test_split
from collections import defaultdict
# os.environ["GOOGLE_API_KEY"] = "AIzaSyB41iEts_InBYR3sHz1bywFYN2JjxlBTJ0"
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# if not GOOGLE_API_KEY:
#     raise ValueError("Please set your GOOGLE_API_KEY environment variable.")


def get_loc_labels(samples_list, sample_metadata_path):
    """
    inputs:
    a path to a samples_list.json, a file containing a list of sample names,
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


def get_top_k_npy_and_save(array, npy_save_path=None, k=512):
    # this has shape (81832 (for half samples split), 4680, 2)
    assert array.shape[2] == 2, f"Input array must have shape (n_rows, n_taxa, 2), and current shape is {array.shape}"
    
    table = array[:, :, 1]
    is_table_populated = (table > 0).astype(int)
    total_populated = is_table_populated.sum()

    top_rows = np.array([
        row[np.argsort(-row[:, 1])[:k]]
        for row in array
    ])
    small_table_populated = (top_rows[:, :, 1] > 0).astype(int).sum()
    print(f"top {k} columns for each row at {npy_save_path}, represents {small_table_populated / total_populated} of original data")
    
    if npy_save_path:
        np.save(npy_save_path, top_rows)
        print(f"Saved {npy_save_path}")

    return top_rows


def save_taxonomy_table(col_names, stacked_data, sample_list, studies_list, save_path, save_name_npy, save_name_cols):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # save study id list into json
    with open(f"{save_path}/studies_list.json", "w") as f1:
        json.dump(studies_list, f1, indent=4)

    with open(f"{save_path}/sample_list.json", "w") as f1:
        json.dump(sample_list, f1, indent=4)

    # Save column names
    with open(os.path.join(save_path, f"{save_name_cols}.json"), "w") as f2:
        json.dump(col_names, f2, indent=4)

    file_name = os.path.join(save_path, f"{save_name_npy}")
    np.save(file_name, stacked_data)
    
    print(f"Saved to {save_path} and numpy array with shape {stacked_data.shape}")


def make_npy_from_df(df, cols_to_drop):
    sample_list = df["sample"].tolist()
    
    df = df.drop(columns=cols_to_drop)
    data = df.to_numpy()
    n_rows, n_cols = data.shape
    col_indices = np.tile(np.arange(n_cols), (n_rows, 1))

    # Get column names
    col_names = df.columns.tolist()
    # Stack the column indices and data along a new third axis.
    stacked_data = np.stack((col_indices, data), axis=2)

    return col_names, stacked_data, sample_list

def split_dataframe_by_samples(df, sample_col="sample", remove_agp=True, split_ratio=0.5):
    """
    Splits the dataframe into two groups, each with roughly {split_ratio} of the total rows assigned to group 1, and others to group 2.
    All samples from the same study (extracted from sample_name) remain together.

    sample_col is in the format "{study_id}_{sample_id}".

    Returns:
      - df_group1: DataFrame for group 1.
      - df_group2: DataFrame for group 2.
      - count1: Number of rows in group 1.
      - count2: Number of rows in group 2.
    """
    # Extract study_id from sample_name.
    # (Assuming sample_name is formatted as "study_id_sample_id")
    df["study_id"] = df[sample_col].apply(lambda x: x.split('_')[0])

    # Get sample count per study.
    study_counts = df["study_id"].value_counts().to_dict()

    # check if remove american gut project
    remove_count = study_counts.get("PRJEB11419", 0)
    print(f"Removing {remove_count} samples due to being in American Gut Project (PRJEB11419).")
    if remove_agp and "PRJEB11419" in study_counts:
        del study_counts["PRJEB11419"]
    # Sort studies by sample count descending.
    sorted_studies = sorted(study_counts.items(), key=lambda x: x[1], reverse=True)

    group1_studies = []
    group2_studies = []
    count1, count2 = 0, 0

    # Greedily assign studies to the group with fewer samples.
    for study, cnt in sorted_studies:
        if count1 < (count1 + count2) * split_ratio:
            group1_studies.append(study)
            count1 += cnt
        else:
            group2_studies.append(study)
            count2 += cnt

    # Create two dataframes based on the assigned studies.
    df_group1 = df[df["study_id"].isin(group1_studies)]
    df_group2 = df[df["study_id"].isin(group2_studies)]

    return df_group1, df_group2, group1_studies, group2_studies


def split_array_by_samples(data_array, sample_ids, locations, remove_agp=True, split_ratio=0.5):
    """
    Splits the data array into two groups by studies (study_id from sample_ids), 
    preserving samples from the same study in the same group.

    Args:
        data_array (np.ndarray): Shape (num_samples, num_taxa, 2)
        sample_ids (List[str]): List of sample IDs, formatted as "studyid_sampleid"
        locations (List[str]): List of locations corresponding to each sample ID
        remove_agp (bool): Whether to remove samples from "PRJEB11419"
        split_ratio (float): Fraction of total rows to aim for in group 1

    Returns:
        - group1_array: np.ndarray for group 1
        - group2_array: np.ndarray for group 2
        - group1_ids: list of sample IDs in group 1
        - group2_ids: list of sample IDs in group 2
        - group1_locs: list of locations for samples in group 1
        - group2_locs: list of locations for samples in group 2
        
    """
    assert len(data_array) == len(sample_ids) == len(locations), "Array and sample_ids and locations must be same length"

    # Extract study_ids
    study_ids = [sid.split('_')[0] for sid in sample_ids]

    # Count samples per study
    study_to_indices = defaultdict(list)
    for idx, study in enumerate(study_ids):
        study_to_indices[study].append(idx)

    # Optionally remove AGP samples
    if remove_agp and "PRJEB11419" in study_to_indices:
        print(f"Removing {len(study_to_indices['PRJEB11419'])} samples due to being in American Gut Project (PRJEB11419).")
        del study_to_indices["PRJEB11419"]

    # Sort studies by number of samples (descending)
    sorted_studies = sorted(study_to_indices.items(), key=lambda x: len(x[1]), reverse=True)

    group1_indices, group2_indices = [], []
    group1_ids, group2_ids = [], []
    group1_locs, group2_locs = [], []
    count1, count2 = 0, 0

    for study, indices in sorted_studies:
        if count1 < (count1 + count2) * split_ratio:
            group1_indices.extend(indices)
            group1_ids.extend([sample_ids[i] for i in indices])
            group1_locs.extend([locations[i] for i in indices])
            count1 += len(indices)
        else:
            group2_indices.extend(indices)
            group2_ids.extend([sample_ids[i] for i in indices])
            group2_locs.extend([locations[i] for i in indices])
            count2 += len(indices)

    group1_array = data_array[group1_indices]
    group2_array = data_array[group2_indices]

    return group1_array, group2_array, group1_ids, group2_ids, group1_locs, group2_locs


def read_taxonomic_table(file_path, split_ratio, nrows = None):
    """takes a file path of a csv file containing the hmc taxonomic table
    returns:
    - a numpy array of data of shape (n_rows, n_cols, 2) where [:, :, 0] represents the index of column and [:, :, 1]
      represents actual table value
    - a list of column names (bacteria names)
    - a list of row names (sample names)
    """
    df = pd.read_csv(file_path, index_col=0, nrows=nrows)
    # 1. shuffle the dataframe
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    # 2. split by sample
    df1, df2, df1_studies, df2_studies = split_dataframe_by_samples(df_shuffled, split_ratio=split_ratio)
    return df1, df2, df1_studies, df2_studies


def main():
    parser = argparse.ArgumentParser(description="Preprocess HMC taxonomic table.")
    parser.add_argument('--taxonomic_table_path', type=str, required=True, help='Path to the taxonomic table CSV file.')
    parser.add_argument('--pretrain_save_dir', type=str, required=True, help='Directory to save pretrain data.')
    parser.add_argument('--finetune_save_dir', type=str, required=True, help='Directory to save finetune data.')
    parser.add_argument('--npy_pretrain_filename', type=str, default="taxonomy_table_pretrain", help='Pretrain .npy file name.')
    parser.add_argument('--npy_finetune_filename', type=str, default="taxonomy_table_finetune", help='Finetune .npy file name.')
    parser.add_argument('--sample_metadata_path', type=str, required=True, help='Finetune .npy file name.')

    parser.add_argument('--split_ratio', type=float, default=0.8, help='Ratio for splitting the dataset into pretrain and finetune sets.')
    parser.add_argument('--nrows', type=int, default=None, help='Number of rows to read from the taxonomic table CSV file.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility.')
    parser.add_argument('--split_finetune_by_study', action='store_true', help='Whether to split finetune data by study.')
    args = parser.parse_args()

    taxonomic_table_path = args.taxonomic_table_path
    pretrain_save_dir = args.pretrain_save_dir
    finetune_save_dir = args.finetune_save_dir
    npy_pretrain_filename = args.npy_pretrain_filename
    npy_finetune_filename = args.npy_finetune_filename
    sample_metadata_path = args.sample_metadata_path
    nrows = args.nrows
    split_ratio = args.split_ratio
    split_finetune_by_study = args.split_finetune_by_study
    
    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"Setting random seed to {args.seed} for reproducibility.")

    pretrain_loc_save_path = os.path.join(pretrain_save_dir, "loc_labels_pretrain.json")
    finetune_loc_save_path = os.path.join(finetune_save_dir, "loc_labels_finetune.json")

    
    print(f"Reading taxonomic table from {taxonomic_table_path} with nrows={nrows} and split_ratio={split_ratio}")
    df1, df2, df1_studies, df2_studies = read_taxonomic_table(taxonomic_table_path, nrows=nrows, split_ratio=split_ratio)
    
    df1_col_names, df1_stacked_data, df1_sample_list = make_npy_from_df(df1, cols_to_drop=['sample', 'study_id'])
    df2_col_names, df2_stacked_data, df2_sample_list = make_npy_from_df(df2, cols_to_drop=['sample', 'study_id'])
    
    save_taxonomy_table(df1_col_names, df1_stacked_data, df1_sample_list, df1_studies, pretrain_save_dir, npy_pretrain_filename, "pretrain_cols")
    save_taxonomy_table(df2_col_names, df2_stacked_data, df2_sample_list, df2_studies, finetune_save_dir, npy_finetune_filename, "finetune_cols")    

    # location labels for pretrain
    pretrain_locations = get_loc_labels(df1_sample_list, sample_metadata_path)
    with open(pretrain_loc_save_path, "w") as f1:
        json.dump(pretrain_locations, f1, indent=4)
        
    # location labels for finetune
    finetune_locations = get_loc_labels(df2_sample_list, sample_metadata_path)
    with open(finetune_loc_save_path, "w") as f1:
        json.dump(finetune_locations, f1, indent=4)

    # train test split for finetune data
    if split_finetune_by_study:
        train_data_arr, test_data_arr, train_samples, test_samples, train_locs, test_locs = split_array_by_samples(df2_stacked_data, df2_sample_list, finetune_locations, split_ratio=0.8)
    else:
        train_data_arr, test_data_arr, train_samples, test_samples, train_locs, test_locs = train_test_split(
            df2_stacked_data, df2_sample_list, finetune_locations, train_size=0.8
        )
    
    # filter top 512
    finetune_traindir = os.path.join(finetune_save_dir, "train")
    finetune_testdir = os.path.join(finetune_save_dir, "test")
    os.makedirs(finetune_traindir, exist_ok=True)
    os.makedirs(finetune_testdir, exist_ok=True)
    
    # save the train and test data arrays
    np.save(os.path.join(finetune_traindir, "finetune_data_train.npy"), train_data_arr)
    np.save(os.path.join(finetune_testdir, "finetune_data_test.npy"), test_data_arr)
    
    # filter and save 512 filtered
    get_top_k_npy_and_save(train_data_arr, os.path.join(finetune_traindir, "finetune_data_train_512.npy"))
    get_top_k_npy_and_save(test_data_arr, os.path.join(finetune_testdir, "finetune_data_test_512.npy"))

    # Save .json files
    with open(os.path.join(finetune_traindir, "finetune_samples_train.json"), "w") as f:
        json.dump(train_samples, f, indent=4)
    with open(os.path.join(finetune_testdir, "finetune_samples_test.json"), "w") as f:
        json.dump(test_samples, f, indent=4)

    with open(os.path.join(finetune_traindir, "finetune_locs_train.json"), "w") as f:
        json.dump(train_locs, f, indent=4)
    with open(os.path.join(finetune_testdir, "finetune_locs_test.json"), "w") as f:
        json.dump(test_locs, f, indent=4)


if __name__ == '__main__':
    main()