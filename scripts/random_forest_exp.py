import numpy as np
import time
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, confusion_matrix
import json
import joblib
from scipy.stats import randint, uniform
import anndata as ad

from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt

import argparse
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

def train_rf(X_train, y_train, search_type):
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

    if search_type == "grid":
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
    
    elif search_type == "random":
        param_distributions = {
            "min_samples_leaf": randint(1, 11),            # integer between 1 and 10
            "max_samples": uniform(0.5, 0.5),              # float between 0.5 and 1.0
            "max_features": uniform(0.1, 0.3),             # float between 0.1 and 0.4
            "n_estimators": randint(200, 1000)               # integer between 50 and 1000
        }
        
        param_distributions  = {"max_features": 0.3446384285364502, "max_samples": 0.8534286719238086, "min_samples_leaf": 3, "n_estimators": 618}

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
    
    elif search_type == "none":
        params = {"max_features": 0.3446384285364502, "max_samples": 0.8534286719238086, "min_samples_leaf": 3, "n_estimators": 618}
        print("Not doing any search, using fixed parameters:", params)
        rf_model.set_params(**params)
        rf_model.fit(X_train, y_train)
        return params, rf_model
    
    else:
        raise ValueError("search_type must be 'grid', 'random', or 'none'")

    # Train the model using grid search
    start_time = time.time()
    search.fit(X_train, y_train)
    end_time = time.time()
    print(f"Time elapsed for search: {(end_time - start_time):.2f} seconds")

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
    print(f"y_probs {y_probs.shape}, y_pred {y_pred.shape}, y_test_binary {y_test_binary.shape}")
    
    # check that the shape of y_probs 1 dimensional
    assert y_probs.ndim == 1, f"y_probs should be 1-dimensional, got {y_probs.ndim} dimensions"
    
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
    
def plot_roc_curve(binary_targets, scores, region, output_dir):
    # Create a new figure for this class
    plt.figure()
    RocCurveDisplay.from_predictions(
        y_true=binary_targets,
        y_pred=scores,
        name=f"ROC: {region}",
        plot_chance_level=True
    )

    plt.title(f"ROC Curve for {region} ({binary_targets.sum()} samples)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(True)

    # Save individual figure
    safe_region = region.replace("/", "_").replace("\\", "_")
    fig_path = os.path.join(output_dir, f"roc_{safe_region}.png")
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved ROC curve for {region} to {fig_path}")

def main():
    parser = argparse.ArgumentParser(description="Script for processing embeddings.")

    parser.add_argument("--train-embed-path", type=str, required=True, help="Path to the anndata where the train embeddings are saved or should be saved in the obsm['embedding'].")
    parser.add_argument("--test-embed-path", type=str, required=True, help="Path to the anndata where the test embeddings are saved or should be saved in the obsm['embedding'].")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the best model and parameters.")
    parser.add_argument("--multiclass", action="store_true", help="multiclass tree or 1 v all trees?")
    parser.add_argument("--do-train", action="store_true", help="train, or just load?")
    parser.add_argument("--search-type", type=str, choices=["grid", "random", "none"], default="random", help="Type of search to perform: 'grid', 'random', or 'none'.")
    parser.add_argument("--target-colname", type=str, default="location", help="Column name in the anndata obs to use as target labels.")
    parser.add_argument("--emb-name", type=str, default="embedding", help="Name of the obsm key where embeddings are stored.")

    args = parser.parse_args()

    train_embed_path = args.train_embed_path
    test_embed_path = args.test_embed_path
    output_dir = args.output_dir
    
    emb_name = args.emb_name
    
    print("args are:", args)
    
    if args.do_train:
        # Load train embeddings and labels
        train_adata = ad.read_h5ad(train_embed_path)
        train_adata = train_adata[train_adata.obs[args.target_colname] != "unknown"]
        X_train = train_adata.obsm[emb_name]
        Y_train = train_adata.obs[args.target_colname]
        print("X_train shape:", X_train.shape)

    # Load test embeddings and labels
    test_adata = ad.read_h5ad(test_embed_path)
    test_adata = test_adata[test_adata.obs[args.target_colname] != "unknown"]
    X_test = test_adata.obsm[emb_name]
    Y_test = test_adata.obs[args.target_colname]    
    
    region_scores = []
    print("X_test shape:", X_test.shape)
    print("about to start training or loading models")
    # this is for 1 v all trees
    if not args.multiclass:
        unique_labels = np.unique(Y_train)
        for region_name in unique_labels:
            if args.do_train:
                print(f"Starting Random Forest classifier on region {region_name}")
                
                # Create binary labels: 1 for current region, 0 otherwise
                y_train_binary = (Y_train == region_name).astype(int)
                y_test_binary = (Y_test == region_name).astype(int)
                best_params, best_model = train_rf(X_train, y_train_binary, args.search_type)

                # Save the best parameters and model for the current region
                save_model(region_name, output_dir, best_params, best_model)
            else:
                print(f"Only doing eval for {region_name}")
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
            best_params, best_model = train_rf(X_train, Y_train, args.search_type)
            save_model("multiclass_tree", output_dir, best_params, best_model)
        else:
            print(f"Only doing eval for all regions")
            best_params, best_model = load_model("multiclass_tree", output_dir)
        
        # Filter test samples with unseen classes BEFORE prediction
        valid_classes = set(best_model.classes_)
        mask_valid = Y_test.isin(valid_classes)

        if not mask_valid.all():
            removed_labels = set(Y_test[~mask_valid].unique())
            print(f"Warning: removing test samples with unseen labels: {removed_labels}")

        X_test_filtered = X_test[mask_valid.values]
        Y_test_filtered = Y_test[mask_valid]
        
        y_probs = best_model.predict_proba(X_test_filtered)
        print("shape of probs and targets is:", y_probs.shape, Y_test_filtered.shape)
        predictions = best_model.predict(X_test_filtered)
        print("types of predictions and targets are:", predictions.dtype, Y_test_filtered.dtype)
        total_accuracy = accuracy_score(Y_test_filtered, predictions)
        conf_mat = confusion_matrix(Y_test_filtered, predictions)
        region_scores = []
        for index, region in enumerate(best_model.classes_):
            scores = y_probs[:, index]
            binary_predictions = (predictions == region).astype(int)
            binary_targets = (Y_test_filtered == region).astype(int)
            region_scores.append(evaluate_binary(region, scores, binary_predictions, binary_targets))
            plot_roc_curve(binary_targets, scores, region, output_dir)
        
        # Save the scores to a file
        conf_row_strs = [str(row) for row in conf_mat]

        region_scores.sort(key=lambda x: x["Region"])
        region_scores.append({"Total Accuracy": total_accuracy,
                            "Categories": list(best_model.classes_),
                            "Confusion Matrix": conf_row_strs})
        os.makedirs(output_dir, exist_ok=True)
        # Save the scores to a file
        scores_file = os.path.join(output_dir, "region_scores.json")
        print(f"Scores for all regions saved to {scores_file}")
        with open(scores_file, "w") as f:
            json.dump(region_scores, f, indent=4)

        print(f"Scores for all regions saved to {scores_file}")
        
if __name__ == "__main__":
    print("Starting random forest experiment script")
    main()
