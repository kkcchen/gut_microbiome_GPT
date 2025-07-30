import numpy as np
import time
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, confusion_matrix
import json
import joblib
from scipy.stats import randint, uniform
import anndata as ad

import argparse
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

def train_rf(X_train, y_train, random_search=True):
    # Initialize the RandomForestClassifier
    rf_model = RandomForestClassifier(
        bootstrap=True,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
    
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    
    n_classes = len(np.unique(y_train))

    if not random_search:
        param_grid = {
            "min_samples_leaf": [1, 10, 100],
            # "max_samples": [1.0, 0.75, 0.5],
            # "max_features": [0.1, 0.2, 0.3],
            "n_estimators": [50, 250, 500, 1000]
        }
        
        # Perform grid search with cross-validation
        search = GridSearchCV(
            estimator=rf_model,
            param_grid=param_grid,
            scoring='roc_auc',
            cv=3,
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
    
    else:
        param_distributions = {
            "min_samples_leaf": randint(1, 11),            # integer between 1 and 10
            "max_samples": uniform(0.5, 0.5),              # float between 0.5 and 1.0
            "max_features": uniform(0.1, 0.3),             # float between 0.1 and 0.4
            "n_estimators": randint(200, 1000)               # integer between 50 and 1000
        }

        # Perform random search with cross-validation
        search = RandomizedSearchCV(
            estimator=rf_model,
            param_distributions=param_distributions,
            n_iter=25,                   # Number of parameter combinations to try
            scoring='roc_auc' if n_classes == 2 else 'f1_weighted',
            cv=3,
            n_jobs=-1,
            random_state=42,
            verbose=1
        )

    # Train the model using grid search
    start_time = time.time()
    search.fit(X_train, y_train)
    end_time = time.time()
    print(f"Time elapsed for grid search: {(end_time - start_time):.2f} seconds")

    # Get the best parameters and model
    return search.best_params_, search.best_estimator_

def save_model(region_name, output_dir, best_params, best_model):
    # Save the best parameters and model for the current region
    region_name = region_name.replace("/", " ")
    region_dir = os.path.join(output_dir, f"{region_name}")
    os.makedirs(region_dir, exist_ok=True)

    # Save best parameters
    with open(os.path.join(region_dir, "best_params.json"), "w") as f:
        json.dump(best_params, f)

    # Save best model
    joblib.dump(best_model, os.path.join(region_dir, "best_model.pkl"))

def load_model(region_name, output_dir):
    # Make region name filesystem-safe
    region_name = region_name.replace("/", " ")
    region_dir = os.path.join(output_dir, f"{region_name}")

    # Load parameters
    with open(os.path.join(region_dir, "best_params.json"), "r") as f:
        best_params = json.load(f)

    # Load model
    best_model = joblib.load(os.path.join(region_dir, "best_model.pkl"))

    return best_params, best_model

def evaluate_binary(region_name, y_probs, y_pred, y_test_binary):
    
    # Check that all shapes are equal
    assert y_probs.shape == y_pred.shape == y_test_binary.shape, \
        f"Shape mismatch: y_probs {y_probs.shape}, y_pred {y_pred.shape}, y_test_binary {y_test_binary.shape}"

    # Check that the shape is (n_samples, 1)
    assert y_probs.ndim == 2 and y_probs.shape[1] == 1, \
        f"Expected shape (n_samples, 1), got {y_probs.shape}"
    
    # Evaluate the model's accuracy on the test set
    n_samples = np.sum(y_test_binary).item()
    accuracy = accuracy_score(y_test_binary, y_pred)
    auc = roc_auc_score(y_test_binary, y_probs)
    average_precision = average_precision_score(y_test_binary, y_probs)
    baseline_precision = np.mean(y_test_binary)

    return {
        "Region": region_name,
        "n_samples": n_samples,
        "Accuracy": accuracy,
        "AUC (ROC)": auc,
        "Average Precision": average_precision,
        "Baseline Precision": baseline_precision
    }

def main():
    parser = argparse.ArgumentParser(description="Script for processing embeddings.")

    parser.add_argument("--train-embed-path", type=str, required=True, help="Path to the anndata where the train embeddings are saved or should be saved in the obsm['embedding'].")
    parser.add_argument("--test-embed-path", type=str, required=True, help="Path to the anndata where the test embeddings are saved or should be saved in the obsm['embedding'].")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the best model and parameters.")
    parser.add_argument("--multiclass", action="store_true", help="multiclass tree or 1 v all trees?")
    parser.add_argument("--do-train", action="store_true", help="train, or just load?")

    args = parser.parse_args()

    train_embed_path = args.train_embed_path
    test_embed_path = args.test_embed_path
    output_dir = args.output_dir
    
    # Load train embeddings and labels
    train_adata = ad.read_h5ad(train_embed_path)
    train_embeddings = train_adata.obsm["embedding"]
    train_loc_labels = train_adata.obs["location"].tolist()

    # Load test embeddings and labels
    test_adata = ad.read_h5ad(test_embed_path)
    test_embeddings = test_adata.obsm["embedding"]
    test_loc_labels = test_adata.obs["location"].tolist()
        
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
    region_scores = []

    # this is for 1 v all trees
    if not args.multiclass:
        for i in range(7):
            if args.do_train:
                print(f"Starting Random Forest classifier on region {i}, which is {list(label_dict.keys())[i]}")
                
                # Create binary labels: 1 for current region, 0 otherwise
                y_train_binary = (Y_train == i).astype(np.int64)
                y_test_binary = (Y_test == i).astype(np.int64)
                best_params, best_model = train_rf(X_train, y_train_binary)

                # Save the best parameters and model for the current region
                region_name = list(label_dict.keys())[i]
                save_model(region_name, output_dir, best_params, best_model)
            else:
                print(f"Only doing eval for {list(label_dict.keys())[i]}")
                best_params, best_model = load_model(region_name, output_dir)
            
            # Predict the labels for the test set using the best model
            y_probs = best_model.predict_proba(X_test)
            assert np.allclose(y_probs.sum(axis=1), 1.0, atol=1e-6), "Not all rows sum to 1"
            y_true_prob = y_probs[:,1]
            y_pred = np.argmax(y_probs, axis=1)
            # Store the scores for the current region
            region_scores.append(evaluate_binary(region_name, y_true_prob, y_pred, y_test_binary))

        # Save the scores to a file
        scores_file = os.path.join(output_dir, "region_scores.json")
        with open(scores_file, "w") as f:
            json.dump(region_scores, f, indent=4)

        print(f"Scores for all regions saved to {scores_file}")
    else:
        if args.do_train:
            print(f"Starting Random Forest classifier on all regions")
            best_params, best_model = train_rf(X_train, Y_train)
            save_model("multiclass_tree", output_dir, best_params, best_model)
        else:
            print(f"Only doing eval for all regions")
            best_params, best_model = load_model("multiclass_tree", output_dir)
        
        y_probs = best_model.predict_proba(X_test)
        print("shape of probs and targets is:", y_probs.shape, Y_test.shape)
        predictions = np.argmax(y_probs, axis=1)
        total_accuracy = accuracy_score(Y_test, predictions)
        conf_mat = confusion_matrix(Y_test, predictions)
        region_scores = []
        for region, index in label_dict.items():
            if region == "unknown":
                continue
            scores = y_probs[:, index]
            binary_predictions = np.array((predictions == index), dtype=int)
            binary_targets = np.array((Y_test == index), dtype=int)            
            region_scores.append(evaluate_binary(region, scores, binary_predictions, binary_targets))
        
        # Save the scores to a file
        conf_row_strs = [str(row) for row in conf_mat]

        region_scores.sort(key=lambda x: x["Region"])
        region_scores.append({"Total Accuracy": total_accuracy,
                            "Categories": list(label_dict.keys()),
                            "Confusion Matrix": conf_row_strs})
        os.makedirs(output_dir, exist_ok=True)
        # Save the scores to a file
        scores_file = os.path.join(output_dir, "region_scores.json")
        print(f"Scores for all regions saved to {scores_file}")
        with open(scores_file, "w") as f:
            json.dump(region_scores, f, indent=4)

        print(f"Scores for all regions saved to {scores_file}")
if __name__ == "__main__":
    main()
