import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import json

import argparse

def main():
    parser = argparse.ArgumentParser(description="Script for processing embeddings.")

    parser.add_argument("--train-embed-path", type=str, required=True, help="Path to the file where the train embeddings are saved or should be saved.")
    parser.add_argument("--train-loc-labels-path", type=str, required=True, help="Path to the train location labels file.")
    parser.add_argument("--test-embed-path", type=str, required=True, help="Path to the file where the test embeddings are saved or should be saved.")
    parser.add_argument("--test-loc-labels-path", type=str, required=True, help="Path to the test location labels file.")

    args = parser.parse_args()

    train_embed_path = args.train_embed_path
    train_loc_labels_path = args.train_loc_labels_path
    test_embed_path = args.test_embed_path
    test_loc_labels_path = args.test_loc_labels_path
    
    # Load train embeddings and labels
    train_embeddings = np.load(train_embed_path)  # (batch_size, emb_dim)
    with open(train_loc_labels_path) as f:
        train_loc_labels = json.load(f)

    # Load test embeddings and labels
    test_embeddings = np.load(test_embed_path)  # (batch_size, emb_dim)
    with open(test_loc_labels_path) as f:
        test_loc_labels = json.load(f)
        
    # Check label lengths match original embeddings
    assert len(train_loc_labels) == train_embeddings.shape[0], f"Mismatch between train labels and embeddings, {len(train_loc_labels)} vs {train_embeddings.shape[0]}"
    assert len(test_loc_labels) == test_embeddings.shape[0], f"Mismatch between test labels and embeddings, {len(test_loc_labels)} vs {test_embeddings.shape[0]}"

    # Location to label mapping
    label_dict = {
        'Australia/New Zealand': 0,
        'Central and Southern Asia': 1,
        'Eastern and South-Eastern Asia': 2,
        'Europe and Northern America': 3,
        'Latin America and the Caribbean': 4,
        'Northern Africa and Western Asia': 5,
        'Sub-Saharan Africa': 6,
        'unknown': 7,
    }

    # Convert train labels
    Y_train = np.array([label_dict[l] for l in train_loc_labels])
    known_train_idx = np.argwhere(Y_train != 7).squeeze(1)
    X_train = train_embeddings[known_train_idx]
    Y_train = Y_train[known_train_idx]

    # Convert test labels
    Y_test = np.array([label_dict[l] for l in test_loc_labels])
    known_test_idx = np.argwhere(Y_test != 7).squeeze(1)
    X_test = test_embeddings[known_test_idx]
    Y_test = Y_test[known_test_idx]
    #
    # # temporary: save the dataset without unknowns
    # # save_dir = "/home/kevin/Desktop/gut_microbiome/dataset/hmc/electra/random_forest/without_unknown"
    # # np.save(os.path.join(save_dir, "taxonomy_table.npy"), X)
    # # np.save(os.path.join(save_dir, "labels.npy"), Y)
    #
    # # here we train the random forest on abundance data with per sample linear normalization
    # X = X[:, :, 1]
    # min_X = np.expand_dims(np.min(X, axis=1), 1)
    # max_X = np.expand_dims(np.max(X, axis=1), 1)
    # X = (X - min_X) / (max_X - min_X + 1e6)
    #
    # instead of doing random forest over all categories, we follow the HMC paper and do per category one-vs-all classification
    for i in range(7):
        print(f"Starting Random Forest classifier on region {i}, which is {list(label_dict.keys())[i]}")
        start_time = time.time()
        
        # Create binary labels: 1 for current region, 0 otherwise
        y_train_binary = (Y_train == i).astype(np.int64)
        y_test_binary = (Y_test == i).astype(np.int64)
    
        # Initialize the RandomForestClassifier.
        rf_model = RandomForestClassifier(
            n_estimators=2000,
            max_features=210,
            min_samples_leaf=1,
            bootstrap=True,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        )
    
        # Train the model on the training data.
        start_time = time.time()
        rf_model.fit(X_train, y_train_binary)
        end_time = time.time()
        print(f"Time elapsed training: {(end_time - start_time):.2f} seconds")
    
        # Predict the labels for the test set.
        y_pred = rf_model.predict(X_test)
    
        # Evaluate the model's accuracy on the test set.
        accuracy = accuracy_score(y_test_binary, y_pred)
        print("Test Accuracy:", accuracy)
        auc = roc_auc_score(y_test_binary, y_pred)
        print("AUC:", auc)
        
if __name__ == "__main__":
    main()
