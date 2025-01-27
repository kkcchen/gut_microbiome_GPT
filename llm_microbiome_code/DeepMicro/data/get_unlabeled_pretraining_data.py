"""
Extend labeled microbiome datasets with unlabeled AGP data.

This script processes labeled microbiome datasets (fruit, vegetable, and IBD)
and extends them with unlabeled data from the American Gut Project (AGP).
It identifies samples in the AGP dataset that are not present in the labeled
datasets, appends them to the labeled data, and saves the extended datasets.

The script performs the following main tasks:
1. Load labeled datasets (fruit, vegetable, IBD) and the AGP dataset.
2. Find unlabeled samples in the AGP dataset not present in the labeled data.
3. Append unlabeled samples to each labeled dataset.
4. Save extended datasets with corresponding labels (-100 for unlabeled samples).

Output files:
- AGP_extended_fruit_all_otu.npy
- AGP_extended_fruit_labels.npy
- AGP_extended_veg_all_otu.npy
- AGP_extended_veg_labels.npy
- AGP_extended_ibd_all_otu.npy
- AGP_extended_ibd_labels.npy

Note: The script assumes that input data are numpy arrays with samples as rows
and features (taxa) as columns.
"""

import numpy as np

all_data = np.load("/path/to/train_encodings_1897.npy")

# Loading the labeled fruit data:
fruit_data = np.load("/path/to/microbiomedata/fruitdata/FRUIT_FREQUENCY_all_otu.npy")
fruit_labels = np.load("/path/to/microbiomedata/fruitdata/FRUIT_FREQUENCY_binary34_labels.npy")[:,0]
# Loading the labeled veg data:
veg_data = np.load("/path/to/microbiomedata/vegdata/VEGETABLE_FREQUENCY_all_otu.npy")
veg_labels = np.load("/path/to/microbiomedata/vegdata/VEGETABLE_FREQUENCY_binary34_labels.npy")[:,0]

# Loading the labeled IBD data:
ibd_data = np.load("/path/to/microbiomedata/IBD_train_otu.npy")
ibd_labels = np.load("/path/to/microbiomedata/total_IBD_label.npy")[:,0]

print(all_data.shape)
print(fruit_data.shape, veg_data.shape, ibd_data.shape)
print(fruit_labels.shape, veg_labels.shape, ibd_labels.shape)


# For each of the fruit, veg and ibd data, we want to get the indices of the samples that are in AGP / all_data but not in the labeled data
# Then we want to get the corresponding samples from the AGP data
# We'll produce six files: "AGP_extended_fruit_all_otu.npy", "AGP_extended_fruit_labels.npy", "AGP_extended_veg_all_otu.npy", "AGP_extended_veg_labels.npy", "AGP_extended_ibd_all_otu.npy", and "AGP_extended_ibd_labels.npy"

# The new data will correspond to the fruit/veg data with unlabeled AGP data appended to the end. The label files will be the same as the original fruit/veg/ibd label files, but with -100 appended to the end for the unlabeled data.


def find_unlabeled_indices(labeled_data, all_data):
    # Assuming labeled_data and all_data are 2D numpy arrays where rows are samples and columns are features (taxa in this case)
    # Convert rows to a set of immutable tuples to enable direct comparison
    labeled_set = set(map(tuple, labeled_data[:,:,0]))
    all_set = set(map(tuple, all_data[:,:,0]))
    
    # Find the difference between the two sets to get rows in all_data not in labeled_data
    unlabeled_set = all_set - labeled_set
    
    # Convert back to list of lists (if necessary) and find indices in all_data
    unlabeled_indices = [i for i, row in enumerate(all_data[:,:,0]) if tuple(row) in unlabeled_set]
    print("len(unlabeled_indices)", len(unlabeled_indices))
    
    return unlabeled_indices

def append_unlabeled_data(labeled_data, labeled_labels, all_data, file_name_otu, file_name_labels):
    unlabeled_indices = find_unlabeled_indices(labeled_data, all_data)
    unlabeled_data = all_data[unlabeled_indices]
    extended_data = np.concatenate((labeled_data, unlabeled_data), axis=0)
    unique, indices = np.unique(extended_data, return_index=True, axis=0)
    print("len(indices)", len(indices), "len(extended_data)", len(extended_data), "len(unique)", len(unique))
    extended_data = extended_data[indices]
    extended_labels = np.concatenate((labeled_labels, np.full(len(unlabeled_data), -100)))
    extended_labels = extended_labels[indices]
    extended_labels = np.expand_dims(extended_labels, axis=-1)
    np.save(file_name_otu, extended_data)
    np.save(file_name_labels, extended_labels)

# For fruit data
append_unlabeled_data(fruit_data, fruit_labels, all_data, "AGP_extended_fruit_all_otu.npy", "AGP_extended_fruit_labels.npy")

# For veg data
append_unlabeled_data(veg_data, veg_labels, all_data, "AGP_extended_veg_all_otu.npy", "AGP_extended_veg_labels.npy")

# For ibd data
append_unlabeled_data(ibd_data, ibd_labels, all_data, "AGP_extended_ibd_all_otu.npy", "AGP_extended_ibd_labels.npy")
