"""
Process and map microbiome data to host subjects.

This script processes microbiome data from various sources, matches it with
filtered data, and creates a mapping of host subjects to their corresponding
microbiome data rows. The script performs the following main tasks:

1. Load and process metadata and sequence data.
2. Load and process filtered microbiome data based on the input argument.
3. Convert and rearrange data to match the format of filtered data.
4. Find matching rows between converted and filtered data.
5. Map run accessions to run IDs and sample information.
6. Create a dictionary mapping host subject IDs to their microbiome data rows.
7. Save the resulting mapping to a pickle file.

Usage:
    python host_to_ids.py <filtered_data_name>

Arguments:
    filtered_data_name: Name of the filtered data file to process. 
                        Options: "total_IBD_otu", "FRUIT_FREQUENCY_all_otu", 
                        or "VEGETABLE_FREQUENCY_all_otu".

Output:
    A pickle file named 'host_to_rows_<filtered_data_name>_512.pkl' containing
    the mapping of host subject IDs to their corresponding microbiome data rows.

Note: This script requires specific input files and data structures to be present
in the working directory and subdirectories.
"""

import pandas as pd
import numpy as np
import pickle
import sys

# Note: metadata files are available at https://files.cqls.oregonstate.edu/David_Lab/microbiome_embeddings/data/AG_new

eer2runid_df = pd.read_csv("err-to-qid.txt", sep="\t")
eer2runid = {eer2runid_df['run_accession'][i]: eer2runid_df['sample_title'][i] for i in range(len(eer2runid_df))}
iddf = pd.read_csv("AG_mapping.txt", sep = '\t')

id2fasta = pd.read_csv("seqs_.07_embed.fasta")
fastas = [id2fasta[::2][:-1].values[i][0] for i in range(len(id2fasta[::2][:-1]))]
fastas2id = {fastas[i]: i for i in range(len(fastas))}
id2fasta = {i: fastas[i] for i in range(len(fastas))}

filtered_data_name = sys.argv[1]

if filtered_data_name == "total_IBD_otu":
    filtered_data = np.load("microbiomedata/" + filtered_data_name + ".npy")
elif filtered_data_name == "FRUIT_FREQUENCY_all_otu":
    filtered_data = np.load("microbiomedata/fruitdata/" + filtered_data_name + ".npy")
elif filtered_data_name == "VEGETABLE_FREQUENCY_all_otu":
    filtered_data = np.load("microbiomedata/vegdata/" + filtered_data_name + ".npy")
else:
    raise ValueError("Filtered data name not recognized")
# filtered_data has a shape of (8571, 1897, 2), with 8571 being the number of samples, 1897 being the maximum number of microbes per sample, and the third dimension haing microbe id as entry 0 and count as entry 1
data = pd.read_csv("seqtab_final_filter.07.txt", sep="\t")

# data.values has a shape of (18480, 26726), with 18480 being the number of samples and 26726 being the total number of microbes in the vocabulary.
# filtered_data is formatted with the values in filtered_data[:,:,0] being the microbe ids and the values in filtered_data[:,:,1] being the counts of each microbe in each sample
# In contrast, data_vals is formatted with a "one-hot" encoding, with each row representing a sample and each column representing one of the 26726 microbes in the vocabulary. The value in data_vals[i,j] is the count of the jth microbe in the ith sample, if the jth microbe is present in the ith sample, and 0 otherwise.

# Convert data_vals to the same format as filtered_data (Note that the true mapping of the microbes to the ids is in fastas2id, and that data_cols are the fasta values, which are not in the same order as the ids)

# Rearrange the columns of data to match the order given by fastas2id
data_cols = data.columns
new_order = []

for i in range(len(fastas)):
    fasta = id2fasta[i]
    if fasta in data_cols:
        new_order.append(fasta)

# Create a new DataFrame with rearranged columns
rearranged_data = data[new_order]

# Update the data variable
data = rearranged_data

# Delete duplicate rows in data
#data = data.drop_duplicates()

#print(f"Number of rows after removing duplicates: {data.shape[0]}")


# Update data_vals to reflect the new order
data_vals = data.values
max_microbes = filtered_data.shape[1]  # Maximum number of microbes per sample in filtered_data
num_samples = data.shape[0]

converted_data = np.zeros((num_samples, max_microbes, 2), dtype=np.int64)

for i in range(num_samples):
    non_zero_indices = np.nonzero(data.values[i])[0]
    num_microbes = min(len(non_zero_indices), max_microbes)
    converted_data[i, :num_microbes, 0] = non_zero_indices[:num_microbes]
    converted_data[i, :num_microbes, 1] = data.values[i, non_zero_indices[:num_microbes]]


# Now converted_data has the same format as filtered_data
# Shape: (num_samples, max_microbes, 2)
# converted_data[:,:,0] contains microbe ids
# converted_data[:,:,1] contains counts of each microbe in each sample

# Now we check which of the rows in converted_data match a row in filtered_data. We check for both matches based on both the ids and counts.
# Function to check if two rows match based on either ids or counts
def rows_match(row1, row2):
    # Sort both rows by microbe ids
    # sorted1 = row1[row1[:, 0].argsort()]
    # sorted2 = row2[row2[:, 0].argsort()]
    
    # # Remove zero-padded entries
    # sorted1 = sorted1[sorted1[:, 1] != 0]
    # sorted2 = sorted2[sorted2[:, 1] != 0]
    
    # Compare the sorted and trimmed rows
    return np.array_equal(row1[:, 0], row2[:, 0]) and np.allclose(row1[:, 1], row2[:, 1], rtol=1e-5, atol=1e-8)


# Initialize a list to store matching indices
matching_indices = []

# Iterate through each row in converted_data
for i, converted_row in enumerate(converted_data):
    # Check if this row matches any row in filtered_data
    for j, filtered_row in enumerate(filtered_data):
        if rows_match(converted_row, filtered_row):
            matching_indices.append((i, j))
            break  # Move to the next converted_row once a match is found

# Print the matching indices
print(f"Found {len(matching_indices)} matches.")

# Generate a list of indices from data that had a match in filtered_data
# using the pandas row index of data
# This should be a list of run accession IDs
matching_data_indices = [data.index[i] for i, _ in matching_indices]

# Now convert from run accessions to run IDs
matching_run_ids = [eer2runid[eer] for eer in matching_data_indices]

# Now select the entries from iddf whose "#SampleID" is in matching_run_ids
matching_samples = iddf[iddf['#SampleID'].isin(matching_run_ids)]

# Delete any duplicates in matching_samples
matching_samples = matching_samples.drop_duplicates()

# Check if the HOST_SUBJECT_ID is unique for each of the matching_samples
unique_hosts = matching_samples['HOST_SUBJECT_ID'].unique()

print(f"Number of unique hosts: {len(unique_hosts)}")
print(f"Number of total samples: {len(matching_samples)}")

if filtered_data_name == "total_IBD_otu":
    filtered_data_512 = np.load("microbiomedata/total_IBD_512.npy")
elif filtered_data_name == "FRUIT_FREQUENCY_all_otu":
    filtered_data_512 = np.load("microbiomedata/fruitdata/FRUIT_FREQUENCY_otu_512.npy")
elif filtered_data_name == "VEGETABLE_FREQUENCY_all_otu":
    filtered_data_512 = np.load("microbiomedata/vegdata/VEGETABLE_FREQUENCY_otu_512.npy")



# Create a dictionary mapping HOST_SUBJECT_ID to rows of filtered_data
host_to_rows = {}
for index, row in matching_samples.iterrows():
    host_id = row['HOST_SUBJECT_ID']
    run_id = row['#SampleID']
    # Find the corresponding index in matching_run_ids
    converted_index = matching_run_ids.index(run_id)
    # Get the corresponding row from filtered_data
    filtered_data_512_row = filtered_data_512[matching_indices[converted_index][1]]
    
    if host_id not in host_to_rows:
        host_to_rows[host_id] = []
    host_to_rows[host_id].append(filtered_data_512_row)

# Save host_to_rows to a pickle file
with open('host_to_rows_' + filtered_data_name + '_512.pkl', 'wb') as f:
    pickle.dump(host_to_rows, f)
