import numpy as np
import json

full_table_path = "/h/chenke83/gut_microbiome_GPT/datasets/finetune/taxonomy_table_finetune.npy"
full_sample_names_path = "/h/chenke83/gut_microbiome_GPT/datasets/finetune/sample_list.json"

save_path = "/h/chenke83/gut_microbiome_GPT/datasets/finetune/test/finetune_data_test.npy"
small_sample_names_path = "/h/chenke83/gut_microbiome_GPT/datasets/finetune/test/finetune_samples_test.json"
top_k_table_path = "/h/chenke83/gut_microbiome_GPT/datasets/finetune/test/finetune_data_test_512.npy"

full_table = np.load(full_table_path)

with open(small_sample_names_path, "r") as f:
    small_sample_names = json.load(f)

with open(full_sample_names_path, "r") as f:
    full_sample_names = json.load(f)

assert len(full_sample_names) == len(full_table)

indices = [full_sample_names.index(name) for name in small_sample_names]
filtered_table = full_table[indices]

print("first check that filtered table, passed through top_k, is the same as the 512 table")
top_k_table = np.load(top_k_table_path)

new_top_k_table = np.array([
    row[np.argsort(-row[:, 1])[:512]]
    for row in filtered_table
])

if not np.array_equal(new_top_k_table[:,:,1], top_k_table[:,:,1]):
    # Find the indices where the tables differ
    difference_indices = np.where(new_top_k_table[:, :, 1] != top_k_table[:, :, 1])

    # Calculate the maximum difference between the tables
    max_difference = np.max(np.abs(new_top_k_table[:, :, 1] - top_k_table[:, :, 1]))

    print("Indices where tables differ:", difference_indices)
    print("Maximum difference between tables:", max_difference)
else:
    print("The new top_k_table matches the original top_k_table.")
    np.save(save_path, filtered_table)
    print("Filtered table shape:", filtered_table.shape)