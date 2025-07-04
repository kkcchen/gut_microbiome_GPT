import numpy as np
import time
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
import json
import joblib
from scipy.stats import randint, uniform

import argparse
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

def main():
    parser = argparse.ArgumentParser(description="Script for processing embeddings.")

    parser.add_argument("--train-embed-path", type=str, required=True, help="Path to the file where the train embeddings are saved or should be saved.")
    parser.add_argument("--train-loc-labels-path", type=str, required=True, help="Path to the train location labels file.")
    parser.add_argument("--test-embed-path", type=str, required=True, help="Path to the file where the test embeddings are saved or should be saved.")
    parser.add_argument("--test-loc-labels-path", type=str, required=True, help="Path to the test location labels file.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the best model and parameters.")

    args = parser.parse_args()

    train_embed_path = args.train_embed_path
    train_loc_labels_path = args.train_loc_labels_path
    test_embed_path = args.test_embed_path
    test_loc_labels_path = args.test_loc_labels_path
    output_dir = args.output_dir
    
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
    # Initialize a dictionary to store scores for each region
    region_scores = []

    for i in range(7):
        print(f"Starting Random Forest classifier on region {i}, which is {list(label_dict.keys())[i]}")
        start_time = time.time()
        
        # Create binary labels: 1 for current region, 0 otherwise
        y_train_binary = (Y_train == i).astype(np.int64)
        y_test_binary = (Y_test == i).astype(np.int64)

        # Initialize the RandomForestClassifier
        rf_model = RandomForestClassifier(
            bootstrap=True,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        )

        # param_grid = {
        #     "min_samples_leaf": [1, 10, 100],
        #     # "max_samples": [1.0, 0.75, 0.5],
        #     # "max_features": [0.1, 0.2, 0.3],
        #     "n_estimators": [50, 250, 500, 1000]
        # }
        
        # # # Perform grid search with cross-validation
        # search = GridSearchCV(
        #     estimator=rf_model,
        #     param_grid=param_grid,
        #     scoring='roc_auc',
        #     cv=3,
        #     n_jobs=-1,
        #     random_state=42,
        #     verbose=1
        # )
        
        param_distributions = {
            "min_samples_leaf": randint(1, 11),            # integer between 1 and 10
            "max_samples": uniform(0.5, 1),              # float between 0.5 and 1.0
            "max_features": uniform(0.1, 0.3),             # float between 0.1 and 0.3
            "n_estimators": randint(200, 1000)               # integer between 50 and 500
        }

        # Perform random search with cross-validation
        search = RandomizedSearchCV(
            estimator=rf_model,
            param_distributions=param_distributions,
            n_iter=25,                   # Number of parameter combinations to try
            scoring='roc_auc',
            cv=3,
            n_jobs=-1,
            random_state=42,
            verbose=1
        )

        # Train the model using grid search
        start_time = time.time()
        search.fit(X_train, y_train_binary)
        end_time = time.time()
        print(f"Time elapsed for grid search: {(end_time - start_time):.2f} seconds")

        # Get the best parameters and model
        best_params = search.best_params_
        best_model = search.best_estimator_

        # Save the best parameters and model for the current region
        region_name = list(label_dict.keys())[i].replace("/", " ")
        region_dir = os.path.join(output_dir, f"{region_name}")
        os.makedirs(region_dir, exist_ok=True)

        # Save best parameters
        with open(os.path.join(region_dir, "best_params.json"), "w") as f:
            json.dump(best_params, f)

        # Save best model
        joblib.dump(best_model, os.path.join(region_dir, "best_model.pkl"))

        # Predict the labels for the test set using the best model
        y_pred = best_model.predict(X_test)

        # Evaluate the model's accuracy on the test set
        n_samples = np.sum(y_test_binary).item()
        accuracy = accuracy_score(y_test_binary, y_pred)
        auc = roc_auc_score(y_test_binary, y_pred)
        average_precision = average_precision_score(y_test_binary, y_pred)
        baseline_precision = np.mean(y_test_binary)

        # Store the scores for the current region
        region_scores.append({
            "Region": list(label_dict.keys())[i],
            "n_samples": n_samples,
            "Accuracy": accuracy,
            "AUC (ROC)": auc,
            "Average Precision": average_precision,
            "Baseline Precision": baseline_precision
        })

    # Save the scores to a file
    scores_file = os.path.join(output_dir, "region_scores.json")
    with open(scores_file, "w") as f:
        json.dump(region_scores, f, indent=4)

    print(f"Scores for all regions saved to {scores_file}")
        
if __name__ == "__main__":
    main()
