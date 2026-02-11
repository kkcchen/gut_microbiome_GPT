import os
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import train_test_split
import yaml
import argparse

import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load and parse the YAML configuration file.
    
    :param config_path: Path to the YAML configuration file.
    :type config_path: Path
    :return: Dictionary containing all configuration parameters.
    :rtype: Dict[str, Any]
    """
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            raise ValueError(f"Error parsing YAML file: {exc}")
    
    # required keys
    required_keys = ['paths', 'general_config', 'downstream_tasks']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration section: {key}")
    
    return config

def extract_paths(config: Dict[str, Any]) -> Dict[str, Path]:
    """
    Extract and convert file paths from config to Path objects.
    
    :param config: Full configuration dictionary.
    :type config: Dict[str, Any]
    :return: Dictionary of Path objects for data and output locations.
    :rtype: Dict[str, Path]
    """
    paths = {}
    for key, value in config['paths'].items():
        paths[key] = Path(value)
        # Optionally validate that input files exist
        if 'save_path' not in key and not paths[key].exists():
            print(f"Warning: Path does not exist: {paths[key]}")
    
    return paths

def read_taxonomic_table(file_path):
    """takes a file path of a csv file containing the hmc taxonomic table
    returns:
    - anndata object with:
        - X: abundance table
        - obs: sample metadata [drr, study_id]
        - var: taxa names
    """
    df = pd.read_csv(file_path)
    df = df.drop(columns=["Unnamed: 0"])
    df = df.set_index("sample")
    # 1. shuffle the dataframe
    df = df.sample(frac=1, random_state=42)
    adata = ad.AnnData(X=df)
    adata.obs["drr"] = [idx.split('_')[1] for idx in adata.obs.index]
    adata.var["taxa"] = adata.var_names
    adata.obs["study_id"] = [idx.split('_')[0] for idx in adata.obs.index]
    return adata

def apply_hmc_filtering(adata: ad.AnnData):
    """Applies the aggressive filtering from the original HMC paper for all data tasks, specifically:
    - remove samples with < 10000 reads (paper reports 16781 removed)
    - remove taxa with < 1000 total reads across remaining samples (paper reports 2018 taxa removed)
    - remove taxa detected in < 100 samples (paper reports 578 taxa removed)
    - remove another round of samples with < 10000 reads across remaining taxa (paper reports 19 removed)
    - remove samples with > 10% reads had unassigned phylum (paper reports 943 removed) 
        - note: domain -> kingdom -> phylum -> class -> order -> family -> genus -> species
    Returns the filtered AnnData object, paper reports 150721 samples and 1422 taxa remain.
    """
    # Step 1: Remove samples with < 10000 reads
    sample_sums = np.array(adata.X.sum(axis=1)).flatten()
    sample_mask = sample_sums >= 10000
    adata = adata[sample_mask].copy()
    print(f"Remove samples with < 10000 reads, total samples removed: {np.sum(~sample_mask)}, total samples remaining: {adata.n_obs}")

    # Step 2: Remove taxa with < 1000 total reads
    taxa_sums = np.array(adata.X.sum(axis=0)).flatten()
    taxa_mask = taxa_sums >= 1000
    adata = adata[:, taxa_mask].copy()
    print(f"Remove taxa with < 1000 total reads, total taxa removed: {np.sum(~taxa_mask)}, total taxa remaining: {adata.n_vars}")

    # Step 3: Remove taxa detected in < 100 samples
    taxa_prevalence = np.array((adata.X > 0).sum(axis=0)).flatten()
    prevalence_mask = taxa_prevalence >= 100
    adata = adata[:, prevalence_mask].copy()
    print(f"Remove taxa detected in < 100 samples, total taxa removed: {np.sum(~prevalence_mask)}, total taxa remaining: {adata.n_vars}")

    # Step 4: Remove samples with < 10000 reads again
    sample_sums = np.array(adata.X.sum(axis=1)).flatten()
    sample_mask = sample_sums >= 10000
    adata = adata[sample_mask].copy()
    print(f"Remove samples with < 10000 reads again, total samples removed: {np.sum(~sample_mask)}, total samples remaining: {adata.n_obs}")

    # Step 5: Remove samples with > 10% unassigned phylum
    # taxa names are in the format of "Bacteria.Actinomycetota.Coriobacteriia.Coriobacteriales.Atopobiaceae.Tractidigestivibacter"
    unknown_phylums = {'NA', 'Incertae Sedis'}
    unknown_phylums_mask = adata.var['taxa'].apply(lambda x: x.split('.')[1] in unknown_phylums)
    unknown_phylum_indices = np.where(unknown_phylums_mask)[0]
    unassigned_counts = np.array(adata.X[:, unknown_phylum_indices].sum(axis=1)).flatten()
    total_counts = np.array(adata.X.sum(axis=1)).flatten()
    unassigned_fraction = unassigned_counts / total_counts
    sample_mask = unassigned_fraction <= 0.1
    adata = adata[sample_mask].copy()
    print(f"Remove samples with > 10% unassigned phylum, total samples removed: {np.sum(~sample_mask)}, total samples remaining: {adata.n_obs}")
    return adata

def get_metadata(samples_list, sample_metadata_path, metadata_fields):
    """
    inputs:
    a list of sample names,
    a path to sample_metadata.tsv, a file containing metadata info of all samples,
    return a dict of metadata fields for the samples in samples_list, each key is a metadata field, each value is a list of values for that field corresponding to samples in samples_list
    """
    metadata_df = pd.read_csv(sample_metadata_path, sep='\t')
    metadata_df["sample_id"] = metadata_df["project"] + "_" + metadata_df["srr"]
    metadata_df = metadata_df.set_index("sample_id")
    metadata_dict = {}
    for sample_id in samples_list:
        row = metadata_df.loc[sample_id]
        for field in metadata_fields:
            if field not in metadata_dict:
                metadata_dict[field] = []
            metadata_dict[field].append(row[field])
    return metadata_dict

def handle_single_task(
    task_name: str, 
    adata: ad.AnnData, 
    tags_df: pd.DataFrame, 
    task_config: Dict[str, Any]
) -> Tuple[ad.AnnData, ad.AnnData]:
    """
    Processes a SINGLE task: subsets the data, handles stratification, 
    and adds task-specific labels. Returns distinct objects to avoid overwriting.
    """
    
    prediction_type = task_config.get('prediction_type', 'classification')
    is_regression = (prediction_type == 'regression')
    reserved_keys = {'prediction_type', 'classification_labels', 'split_strategy'}
    
    class_map = {}
    if not is_regression and 'classification_labels' in task_config:
        for label_group in task_config['classification_labels']:
            for target_label, raw_variants in label_group.items():
                for variant in raw_variants:
                    class_map[str(variant)] = target_label

    # Containers for INDICES (not the data itself yet)
    train_indices = []
    test_indices = []
    
    # Global map for this task: { index : raw_value }
    task_raw_labels = {}

    for study_id, study_config in task_config.items():
        if study_id in reserved_keys:
            continue
            
        tag_col = study_config.get('tag_column')
        if not tag_col:
            continue

        # Filter metadata for this study + tag
        study_tags = tags_df[
            (tags_df['project'] == study_id) & 
            (tags_df['tag'] == tag_col)
        ]
        
        if study_tags.empty:
            continue

        # Map: Sample_ID -> Raw_Value
        sample_to_label = dict(zip(study_tags['srr'], study_tags['value']))
        
        # Find intersection with AnnData
        mask = adata.obs['drr'].isin(sample_to_label.keys())
        valid_indices = adata.obs.index[mask].tolist()
        
        if not valid_indices:
            continue

        # Store labels for these indices for later assignment
        current_labels = adata.obs.loc[valid_indices, 'drr'].map(sample_to_label)
        task_raw_labels.update(current_labels.to_dict())

        is_train = study_config.get('is_train', False)
        both_train_test = study_config.get('both_train_test', False)

        if both_train_test:
            # STRATIFICATION
            labels_for_split = current_labels.values
            
            # Map labels BEFORE stratifying to ensure balanced classes
            if not is_regression and class_map:
                labels_for_split = pd.Series(labels_for_split).astype(str).map(class_map).fillna('__unknown__')
            
            try:
                # 80/20 Stratified Split
                tr_idx, te_idx = train_test_split(
                    valid_indices, 
                    test_size=0.2, 
                    stratify=labels_for_split if not is_regression else None
                )
            except ValueError:
                # Fallback if class size < 2
                tr_idx, te_idx = train_test_split(valid_indices, test_size=0.2)

            train_indices.extend(tr_idx)
            test_indices.extend(te_idx)
            
        else:
            # Deterministic Assignment
            if is_train:
                train_indices.extend(valid_indices)
            else:
                test_indices.extend(valid_indices)

    def create_subset(indices: List[str], subset_type: str):
        # Unique indices only (within this task)
        indices = list(set(indices))
        
        if not indices:
            return None # Handle empty case gracefully later

        subset = adata[indices].copy()
        
        # Add Task Metadata
        subset.obs['downstream_task'] = task_name
        subset.obs['categorical_label'] = np.nan
        subset.obs['continuous_label'] = np.nan
        
        # Map labels
        raw_vals = subset.obs.index.map(task_raw_labels)
        
        if is_regression:
            subset.obs['continuous_label'] = pd.to_numeric(raw_vals, errors='coerce')
            subset = subset[subset.obs['continuous_label'].notna()].copy()
        else:
            if class_map:
                subset.obs['categorical_label'] = pd.Series(raw_vals, index=subset.obs.index).astype(str).map(class_map)
            else:
                subset.obs['categorical_label'] = raw_vals
            
            subset = subset[subset.obs['categorical_label'].notna()].copy()
            
        return subset

    adata_train = create_subset(train_indices, "Train")
    adata_test = create_subset(test_indices, "Test")

    n_train = adata_train.n_obs if adata_train else 0
    n_test = adata_test.n_obs if adata_test else 0
    print(f"Processed Task: {task_name:<20} | Train: {n_train:<5} | Test: {n_test:<5}")
    
    return adata_train, adata_test

def build_multitask_dataset(
    adata: ad.AnnData, 
    tags_df: pd.DataFrame, 
    full_config: Dict[str, Any]
) -> Tuple[ad.AnnData, ad.AnnData]:
    """
    Master function to process all tasks and concatenate them.
    """
    all_train_adatas = []
    all_test_adatas = []

    # Loop through the tasks (e.g., 'sex', 'bmi')
    # Assumes full_config is exactly the structure: {'sex': {...}, 'bmi': {...}}
    for task_name, task_config in full_config.items():
        if task_name == 'tasks': # Handle if user passed the root 'tasks' key
            continue
            
        tr, te = handle_single_task(task_name, adata, tags_df, task_config)
        
        if tr is not None and tr.n_obs > 0:
            all_train_adatas.append(tr)
        if te is not None and te.n_obs > 0:
            all_test_adatas.append(te)
    
    print("-" * 60)
    print("Concatenating datasets...")
    
    final_train = ad.concat(all_train_adatas, join='outer') if all_train_adatas else None
    final_test = ad.concat(all_test_adatas, join='outer') if all_test_adatas else None
    
    if final_train:
        print(f"Final Combined Train: {final_train.n_obs} samples across {len(all_train_adatas)} tasks.")
        
    if final_test:
        print(f"Final Combined Test:  {final_test.n_obs} samples across {len(all_test_adatas)} tasks.")

    return final_train, final_test

def print_adata_summary(name, ad_obj):
    print(f"\n{'='*20} {name} {'='*20}")
    if ad_obj is None:
        print("Object is None")
        return

    print(f"Shape: {ad_obj.shape} (Samples x Features)")
    print(f"Obs Columns: {ad_obj.obs.columns.tolist()}")

def print_task_counts(name, ad_obj):
    if ad_obj is None or 'downstream_task' not in ad_obj.obs:
        return

    print(f"\n--- {name} Task Breakdown ---")
    tasks = ad_obj.obs['downstream_task'].unique()
    
    for task in tasks:
        subset = ad_obj[ad_obj.obs['downstream_task'] == task]
        print(f"\nTask: {task} (Total: {subset.n_obs})")
        
        # Check if regression (continuous) or classification (categorical)
        # We check the count of non-NaN values
        n_cont = subset.obs['continuous_label'].notna().sum()
        n_cat = subset.obs['categorical_label'].notna().sum()
        
        if n_cont > 0:
            print("  Type: Regression")
            # For regression, useful stats are min/max/mean rather than value counts
            stats = subset.obs['continuous_label'].describe()[['min', 'max', 'mean', 'std']]
            print(f"  Stats:\n{stats.to_string()}")
        elif n_cat > 0:
            print("  Type: Classification")
            counts = subset.obs['categorical_label'].value_counts()
            print(f"  Class Counts:\n{counts.to_string()}")
        else:
            print("  (No valid labels found)")
        

def preprocess(config_path: Path):
    """
    Main preprocessing workflow for the Human Microbiome Compendium (HMC) data. Doesn't return anything, just saves the processed data to disk.
    The saved files are:
    - pretrain.h5ad: preprocessed data for pretraining
    - downstream_train.h5ad: preprocessed data for downstream task training
    - downstream_test.h5ad: preprocessed data for downstream task testing
    These files are set up in a way that smoothly hooks into the train/eval/test pipeline. Notably, there are redundant samples in the downstream
    files. This is intentional for ease of use. If you want a different split or define more tasks, feel free to modify to config accordingly.
    
    :param config_path: Path to the YAML configuration file.
    :type config_path: Path
    """
    # handle yaml config
    config = load_config(config_path)
    paths = extract_paths(config)
    general_config = config['general_config']
    downstream_tasks = config['downstream_tasks']
    ## seed
    np.random.seed(general_config['seed'])

    # 1. preprocessing of original data, handles loading, reading, filtering
    adata = read_taxonomic_table(paths['taxonomic_table_path'])
    adata = apply_hmc_filtering(adata)

    # 2. add metadata
    metadata_fields = general_config['metadata_fields']
    metadata_dict = get_metadata(adata.obs.index.tolist(), paths['metadata_path'], metadata_fields)
    for field in metadata_fields:
        adata.obs[field] = metadata_dict[field]

    # 3. handle downstream task splits
    # 3.1 handle all downstream tasks without location
    tags_df = pd.read_csv(paths['tags_path'], sep='\t')
    final_train_adata, final_test_adata = build_multitask_dataset(
        adata, 
        tags_df, 
        downstream_tasks
    )

    # 3.2 add location as a downstream task if specified
    if 'location' in config:
        print("\nAdding 'location' downstream task based on 'region' metadata...")
        location_config = config['location']
        train_studies = [study for study, cfg in location_config.items() 
                    if cfg.get('is_train') == True]
        test_studies = [study for study, cfg in location_config.items() 
                    if cfg.get('is_train') == False]
        print("Checking for multi-region studies in location config...")
        for study in train_studies + test_studies:
            study_data = adata[adata.obs['study_id'] == study]
            regions = study_data.obs['region'].unique()
            if len(regions) > 1:
                print(f"\n{study} has MULTIPLE regions:")
                print(study_data.obs['region'].value_counts())
        train_location = adata[adata.obs['study_id'].isin(train_studies)].copy()
        test_location = adata[adata.obs['study_id'].isin(test_studies)].copy()
        train_location.obs['downstream_task'] = 'location'
        train_location.obs['categorical_label'] = train_location.obs['region']
        train_location.obs['continuous_label'] = None
        
        test_location.obs['downstream_task'] = 'location'
        test_location.obs['categorical_label'] = test_location.obs['region']
        test_location.obs['continuous_label'] = None
        
        # Verify no overlap
        train_study_set = set(train_location.obs['study_id'].unique())
        test_study_set = set(test_location.obs['study_id'].unique())
        overlap = train_study_set & test_study_set
        
        if overlap:
            print(f"WARNING: Studies in both train and test: {overlap}")
        else:
            print(f"No study overlap between train and test")
        final_train_adata = ad.concat([final_train_adata, train_location], join='outer')
        final_test_adata = ad.concat([final_test_adata, test_location], join='outer')

    
    # 3.3 remove any unique samples in final_train and final_test from the original adata to avoid data leakage
    used_indices = set()
    if final_train_adata is not None:
        used_indices.update(final_train_adata.obs.index.unique())

    if final_test_adata is not None:
        used_indices.update(final_test_adata.obs.index.unique())

    print(f"Total unique samples used in downstream tasks: {len(used_indices)}")
    adata = adata[~adata.obs.index.isin(used_indices)].copy()
    # 3.4 withhold any studies specified in config withhold_studies
    if 'withhold_studies' in general_config:
        studies_to_withhold = general_config['withhold_studies']
        mask = adata.obs['study_id'].isin(studies_to_withhold)
        withheld_samples = adata[mask].copy()
        adata = adata[~mask].copy()
        print(f"Withheld {withheld_samples.n_obs} samples from studies: {studies_to_withhold}")
    # 4. output summary statistics
    print_adata_summary("Pretrain Adata", adata)
    print_adata_summary("Downstream Train Adata", final_train_adata)
    print_adata_summary("Downstream Test Adata", final_test_adata)
    print_task_counts("Train", final_train_adata)
    print_task_counts("Test", final_test_adata)
    # 5. save processed data
    pretrain_save_path = os.path.join(paths['save_path'], 'pretrain.h5ad')

    os.makedirs(paths['save_path'], exist_ok=True)

    adata.write_h5ad(pretrain_save_path)
    print(f"Pretrain data saved to: {pretrain_save_path}")
    downstream_train_save_path = os.path.join(paths['save_path'], 'downstream_train.h5ad')
    if final_train_adata is not None:
        final_train_adata.write_h5ad(downstream_train_save_path)
        print(f"Downstream Train data saved to: {downstream_train_save_path}")

    downstream_test_save_path = os.path.join(paths['save_path'], 'downstream_test.h5ad')
    if final_test_adata is not None:
        final_test_adata.write_h5ad(downstream_test_save_path)
        print(f"Downstream Test data saved to: {downstream_test_save_path}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess gut microbiome data based on configuration.")
    parser.add_argument("--config", type=str, required=True, help="Path to the preprocessing configuration YAML file.")
    args = parser.parse_args()
    config_path = Path(args.config)
    preprocess(config_path)