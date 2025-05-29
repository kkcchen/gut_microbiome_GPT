import numpy as np
import pandas as pd
import os
# import fasttext
import json
import argparse
# os.environ["GOOGLE_API_KEY"] = "AIzaSyB41iEts_InBYR3sHz1bywFYN2JjxlBTJ0"
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# if not GOOGLE_API_KEY:
#     raise ValueError("Please set your GOOGLE_API_KEY environment variable.")


def get_loc_labels(samples_list_path, sample_metadata_path):
    """
    inputs:
    a path to a samples_list.json, a file containing a list of sample names,
    a path to sample_metadata.tsv, a file containing location info of all samples,
    return a list of locations corresponding to the samples
    """
    with open(samples_list_path) as f:
        samples_list = json.load(f)
    metadata_df = pd.read_csv(sample_metadata_path, sep='\t')
    metadata_df["sample_id"] = metadata_df["project"] + "_" + metadata_df["srr"]
    metadata_df = metadata_df.set_index("sample_id")
    location_list = []
    for sample_id in samples_list:
        row = metadata_df.loc[sample_id]
        location_list.append(row.region)
    return location_list


def train_test_split(npy_file_path, save_path, test_size=0.2, random_state=42):
    """
    Splits a NumPy array into training and testing arrays along the first axis.

    Parameters:
      arr (np.ndarray): Input array of shape (N, ...).
      test_size (float): Fraction of rows to allocate to the test set.
      random_state (int, optional): Seed for reproducibility.

    Returns:
      train (np.ndarray): Training array.
      test (np.ndarray): Testing array.
    """
    if random_state is not None:
        np.random.seed(random_state)
    arr = np.load(npy_file_path)

    print('Original shape: {}'.format(arr.shape))
    # Shuffle the row indices
    indices = np.arange(arr.shape[0])
    np.random.shuffle(indices)

    # Calculate the splitting index
    split_idx = int(arr.shape[0] * (1 - test_size))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    # Split the array
    train = arr[train_indices]
    test = arr[test_indices]
    print('Train shape: {}'.format(train.shape))
    print('Test shape: {}'.format(test.shape))
    base_name = os.path.splitext(os.path.basename(npy_file_path))[0]
    train_save_name = os.path.join(save_path, f"{base_name}_train.npy")
    test_save_name = os.path.join(save_path, f"{base_name}_test.npy")
    np.save(train_save_name, train)
    np.save(test_save_name, test)



def get_top_k_npy(npy_file_path, npy_save_path, k=512):
    data = np.load(npy_file_path)
    # this has shape (81832, 4680, 2)
    table = data[:, :, 1]
    is_table_populated = (table > 0).astype(int)
    total_populated = is_table_populated.sum()

    top_rows = np.array([
        row[np.argsort(-row[:, 1])[:k]]
        for row in data
    ])
    small_table_populated = (top_rows[:, :, 1] > 0).astype(int).sum()
    file_name = os.path.join(npy_save_path, f"taxonomy_table_{k}.npy")
    np.save(file_name, top_rows)
    print(f"Saved {file_name}")
    print(f"top {k} columns for each row, represents {small_table_populated / total_populated} of original data")

    return file_name




def save_taxonomy_table(df_to_save, save_path, save_name, cols_to_drop):
    sample_list = df_to_save["sample"]
    with open(f"{save_path}/sample_list.json", "w") as f1:
        json.dump(sample_list.tolist(), f1)

    df = df_to_save.drop(columns=cols_to_drop)
    data = df.to_numpy()
    n_rows, n_cols = data.shape
    col_indices = np.tile(np.arange(n_cols), (n_rows, 1))

    # Save column names
    col_names = df.columns.tolist()
    with open(os.path.join(save_path, f"{save_name}_colnames.json"), "w") as f2:
        json.dump(col_names, f2)

    # Stack the column indices and data along a new third axis.
    stacked_data = np.stack((col_indices, data), axis=2)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_name = os.path.join(save_path, f"{save_name}.npy")
    np.save(file_name, stacked_data)
    print(f"Saved {file_name}")
    print(f"Saved column names to {save_name}_colnames.json")
    print(f"saved numpy array has shape {stacked_data.shape}")


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
    if remove_agp:
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


def read_taxonomic_table(file_path, df1_save_path, df2_save_path, nrows = None, split_ratio=0.5):
    """takes a file path of a csv file containing the hmc taxonomic table
    returns:
    - a numpy array of data of shape (n_rows, n_cols, 2) where [:, :, 0] represents the index of column and [:, :, 1]
      represents actual table value
    - a list of column names (bacteria names)
    - a list of row names (sample names)
    """
    df = pd.read_csv(file_path, index_col=0, nrows=nrows)
    col_names = df.columns.tolist()
    # 1. shuffle the dataframe
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    # 2. split by sample
    df1, df2, df1_studies, df2_studies = split_dataframe_by_samples(df_shuffled, split_ratio=split_ratio)

    # save df1
    if not os.path.exists(df1_save_path):
        os.makedirs(df1_save_path)
    # save taxonomy table
    save_taxonomy_table(df1, df1_save_path, "taxonomic_table_pretrain", ['sample', 'study_id'])
    # save study id list into json
    with open(f"{df1_save_path}/studies_list.json", "w") as f1:
        json.dump(df1_studies, f1)

    # save df2
    if not os.path.exists(df2_save_path):
        os.makedirs(df2_save_path)
    # save taxonomy table
    save_taxonomy_table(df2, df2_save_path, "taxonomic_table_finetune", ['sample', 'study_id'])
    # save study id list into json
    with open(f"{df2_save_path}/studies_list.json", "w") as f2:
        json.dump(df2_studies, f2)

    return col_names


# def create_vocab_embeddings_biowordvec(col_names, npy_save_dir, batch_size=256):
#     """
#     takes in a list of column names, generates an embedding for each column name
#     use biowordvec embeddings
#     GOOGLE GEMINI API KEY: AIzaSyB41iEts_InBYR3sHz1bywFYN2JjxlBTJ0
#     """
#     # Replace the file path with the location of your downloaded BioWordVec model.
#     model_path = "/home/kevin/Desktop/gut_microbiome/pretrained_models/BioWordVec_PubMed_MIMICIII_d200.bin"

#     bio_model = fasttext.load_model(model_path)

#     # Retrieve the vector for a word (works even if the word is OOV due to subword features)
#     words = [bacteria.split(".")[-1] for bacteria in col_names]
#     all_embeddings = []
#     for i in range(0, len(words), batch_size):
#         batch = words[i : i + batch_size]
#         batch_embeddings = [bio_model.get_word_vector(word) for word in batch]
#         all_embeddings.extend(batch_embeddings)

#     embedding = np.array(all_embeddings)
#     print("Embedding shape:", embedding.shape)
#     if not os.path.exists(npy_save_dir):
#         os.makedirs(npy_save_dir)
#     save_name = os.path.join(npy_save_dir, "bac_vocab.npy")
#     np.save(save_name, embedding)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess HMC taxonomic table.")
    parser.add_argument('--taxonomic_table_path', type=str, required=True, help='Path to the taxonomic table CSV file.')
    parser.add_argument('--pretrain_save_dir', type=str, required=True, help='Directory to save pretrain data.')
    parser.add_argument('--finetune_save_dir', type=str, required=True, help='Directory to save finetune data.')
    parser.add_argument('--npy_pretrain_file', type=str, default="taxonomy_table_pretrain.npy", help='Pretrain .npy file name.')
    parser.add_argument('--npy_finetune_file', type=str, default="taxonomy_table_finetune.npy", help='Finetune .npy file name.')
    parser.add_argument('--split_ratio', type=float, default=0.75, help='Ratio for splitting the dataset into pretrain and finetune sets.')
    parser.add_argument('--nrows', type=int, default=None, help='Number of rows to read from the taxonomic table CSV file.')
    args = parser.parse_args()

    taxonomic_table_path = args.taxonomic_table_path
    pretrain_save_dir = args.pretrain_save_dir
    finetune_save_dir = args.finetune_save_dir
    npy_pretrain_file = args.npy_pretrain_file
    npy_finetune_file = args.npy_finetune_file
    nrows = args.nrows
    split_ratio = args.split_ratio

    # npy1_512_path = "/home/kchen/microbiome/gut_microbiome_GPT/datasets/hmc/electra/transformer/taxonomy_table_512.npy"

    print(f"Reading taxonomic table from {taxonomic_table_path} with nrows={nrows} and split_ratio={split_ratio}")
    col_names = read_taxonomic_table(taxonomic_table_path, pretrain_save_dir, finetune_save_dir, nrows=nrows, split_ratio=split_ratio)
    # vocab_embeddings = create_vocab_embeddings_biowordvec(col_names, npy_save_dir)
    top_512_pretrain = get_top_k_npy(os.path.join(pretrain_save_dir, npy_pretrain_file), pretrain_save_dir)
    top_512_finetune = get_top_k_npy(os.path.join(finetune_save_dir, npy_finetune_file), finetune_save_dir)
    train_test_split(top_512_pretrain, pretrain_save_dir)

    # rf_sample_list_path = "/home/kevin/Desktop/gut_microbiome/dataset/hmc/electra/random_forest/sample_list.json"
    # sample_metadata_path = "/home/kevin/Desktop/gut_microbiome/dataset/hmc/sample_metadata.tsv"
    # locations = get_loc_labels(rf_sample_list_path, sample_metadata_path)
    # label_save_path = "/home/kevin/Desktop/gut_microbiome/dataset/hmc/electra/random_forest/labels.json"
    # with open(label_save_path, "w") as f1:
    #     json.dump(locations, f1)

    # rf_npy_path = "/home/kevin/Desktop/gut_microbiome/dataset/hmc/electra/random_forest/taxonomy_table.npy"
    # rf_save_path =



