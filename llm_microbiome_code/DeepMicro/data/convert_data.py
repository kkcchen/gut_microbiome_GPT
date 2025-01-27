"""
This script converts microbiome data from a numpy array format to one-hot encoded abundance tables.
It processes the input data, normalizes abundances, and saves the results as CSV files.

The script performs the following main tasks:
1. Loads microbiome data and labels from numpy files
2. Converts the data into one-hot encoded abundance tables
3. Normalizes the abundances to sum to one for each sample
4. Saves the processed abundances and labels as CSV files

Usage:
python convert_data.py [--data_path DATA_PATH] [--data_name DATA_NAME] 
                       [--labels_name LABELS_NAME] [--train_encodings_path TRAIN_ENCODINGS_PATH]

"""

import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default="/path/to/microbiomedata/")
parser.add_argument('--data_name', type=str, default="schirmer_IBD_otu")
parser.add_argument('--labels_name', type=str, default="schirmer_IBD_labels")
parser.add_argument('--train_encodings_path', type=str, default="/path/to/train_encodings_folder/")
args = parser.parse_args()

data = np.load(args.data_path + args.data_name + ".npy")
o_ibd_data = np.load(args.train_encodings_path + "train_encodings_1897.npy")
data_labels = np.load(args.data_path + args.labels_name + ".npy")

data_abundances = data[:, :, 1]

# Need to know the highest vocab entry
max_vocab = int(np.max(o_ibd_data[:, :, 0]))

# Convert ibd data into abundance tables
one_hot_abundances = np.zeros((data.shape[0], max_vocab+1))
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        one_hot_abundances[i, int(data[i, j, 0])] = data_abundances[i, j]

# Next, normalize the abundances to sum to one
total_abundance = np.sum(one_hot_abundances, axis=1)

for i in range(one_hot_abundances.shape[0]):
    one_hot_abundances[i, :] /= total_abundance[i] if total_abundance[i] > 0 else 1



# save the abundances and labels as csv files
# save with no more than 4 significant figures for the abundance and as ints for the labels
one_hot_abundances_rounded = np.around(one_hot_abundances, decimals=4)
np.savetxt('oha_' + args.data_name + '.csv', one_hot_abundances_rounded.reshape(data.shape[0], -1), delimiter=',', fmt='%g')
np.savetxt('oha_' + args.labels_name + '.csv', data_labels.astype(int), delimiter=',', fmt='%i')


