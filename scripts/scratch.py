import numpy as np

file_path = "/home/kevin/Desktop/gut_microbiome/dataset/llm_microbiome/vocab_embeddings.npy"
np_array = np.load(file_path)
print(np_array.shape)