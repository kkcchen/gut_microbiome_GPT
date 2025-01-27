"""
This script merges 16S rRNA sequences with their corresponding embeddings.

It performs the following tasks:
1. Loads a pre-trained embedding matrix from a PyTorch file.
2. Reads 16S rRNA sequences from a text file.
3. Matches each sequence with its corresponding embedding.
4. Formats the embeddings as space-separated floating-point values.
5. Writes the merged data (sequence + embedding) to a new output file.

The resulting file contains each sequence followed by its embedding values,
which can be used with the R script to calculate correlations.
"""

import torch



# Load the embedding matrix
# The embedding matrix must be in the same order as the sequences in the seqs_07.txt file,
# with one embedding per 16S sequence.
embedding_matrix = torch.load('epoch_120_IBD_avg_vocab_embeddings.pth')[:,:,0]

# Initialize a list to hold the merged data
merged_data = []

# Open the sequence file and process it
with open('/path/to/seqs_.07_embed.txt', 'r') as seq_file:
    for line_number, line in enumerate(seq_file):
        # Every other line starting from 0 contains a sequence ID, so we skip those
        if line_number % 2 != 0:
            sequence = line.strip()
            # Assuming the sequence directly maps to an index in the embedding matrix
            # This might need adjustment based on how sequences are related to embeddings
            embedding_index = (line_number - 1) // 2
            embedding_values = embedding_matrix[embedding_index].tolist()
            # Create a space-separated string of embedding values
            formatted_embedding_str = ' '.join(f"{val:.6f}" for val in embedding_values)
            # Append the sequence and its embedding to the merged data list
            merged_data.append(f"{sequence} {formatted_embedding_str}")

# Write the merged data to a new file
with open('merged_sequences_embeddings.txt', 'w') as output_file:
    for data in merged_data:
        output_file.write(data + '\n')