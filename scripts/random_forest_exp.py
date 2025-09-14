import numpy as np
import time
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, confusion_matrix
import json
import joblib
from scipy.stats import randint, uniform
import anndata as ad
import matplotlib.pyplot as plt

import argparse
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from trainers.test_functions import evaluate_multiclass_and_save, evaluate_regression_and_save

def train_rf(X_train, y_train, search_type, regression=False):
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    
    if regression:
        rf_model = RandomForestRegressor(
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )
        search_scoring = 'neg_mean_squared_error'
    else:
        n_classes = len(np.unique(y_train))
        rf_model = RandomForestClassifier(
            bootstrap=True,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        )
        search_scoring = 'roc_auc' if n_classes == 2 else 'f1_weighted'

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
            scoring=search_scoring,
            cv=3,
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
    
    elif search_type == "random":
        param_distributions = { # discrete for min_samples_leaf?
            "min_samples_leaf": randint(2, 20),
            "max_samples": uniform(0.5, 0.5),
            "max_features": uniform(0.5, 0.5),
            "n_estimators": randint(200, 1000)
        }
        
        # Perform random search with cross-validation
        search = RandomizedSearchCV(
            estimator=rf_model,
            param_distributions=param_distributions,
            n_iter=10,                   # Number of parameter combinations to try
            scoring=search_scoring,
            cv=3,
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
    
    elif search_type == "none":
        params = {"max_features": 0.6872700594236812, "max_samples": 0.8534286719238086, "min_samples_leaf": 3, "n_estimators": 861}
        print("Not doing any search, using fixed parameters:", params)
        rf_model.set_params(**params)
        rf_model.fit(X_train, y_train)
        return params, rf_model
    
    else:
        raise ValueError("search_type must be 'grid', 'random', or 'none'")

    # Train the model using search
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

def model_exists(region_name, output_dir):
    region_name = region_name.replace("/", " ")
    region_dir = os.path.join(output_dir, f"{region_name}")
    return os.path.exists(region_dir) and os.path.exists(os.path.join(region_dir, "best_model.pkl")) and os.path.exists(os.path.join(region_dir, "best_params.json"))
    
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


def main():
    parser = argparse.ArgumentParser(description="Script for processing embeddings.")

    parser.add_argument("--train-embed-path", type=str, required=True, help="Path to the anndata where the train embeddings are saved or should be saved in the obsm['embedding'].")
    parser.add_argument("--test-embed-path", type=str, required=True, help="Path to the anndata where the test embeddings are saved or should be saved in the obsm['embedding'].")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the best model and parameters.")
    parser.add_argument("--model-type", type=str, choices=["multiclass", "one-vs-all", "regression"], required=True, help="Type of model to train: 'multiclass' for one model handling all classes, 'one-vs-all' for one model per class, or 'regression' for regression tasks.")
    parser.add_argument("--search-type", type=str, choices=["grid", "random", "none"], default="random", help="Type of search to perform: 'grid', 'random', or 'none'.")
    parser.add_argument("--target-colname", type=str, default="location", help="Column name in the anndata obs to use as target labels.")
    parser.add_argument("--emb-name", type=str, default="embedding", help="Name of the obsm key where embeddings are stored.")
    parser.add_argument("--ignored-labels", type=str, nargs='*', default=["unknown"], help="List of labels to ignore in the target column.")
    parser.add_argument("--downstream-task", type=str, required=True, help="Type of downstream task to perform: 'classification' or 'regression'.")

    args = parser.parse_args()

    train_embed_path = args.train_embed_path
    test_embed_path = args.test_embed_path
    output_dir = args.output_dir
    
    emb_name = args.emb_name
        
    print("args are:", args)
    
    # Load train embeddings and labels
    train_adata = ad.read_h5ad(train_embed_path)
    if args.downstream_task == "location":
        assert "unknown" in args.ignored_labels, "For location task, 'unknown' must be in ignored labels."
    train_adata = train_adata[train_adata.obs['downstream_task'] == args.downstream_task]
    train_adata = train_adata[~train_adata.obs[args.target_colname].isin(args.ignored_labels)]
    X_train = train_adata.obsm[emb_name]
    Y_train = train_adata.obs[args.target_colname]
    
    print("X_train shape:", X_train.shape)
    print("Y_train shape:", Y_train.shape)

    # Load test embeddings and labels
    test_adata = ad.read_h5ad(test_embed_path)
    if args.downstream_task != "location":
        test_adata = test_adata[test_adata.obs['downstream_task'] == args.downstream_task]        
    test_adata = test_adata[~test_adata.obs[args.target_colname].isin(args.ignored_labels)]
    X_test = test_adata.obsm[emb_name]
    Y_test = test_adata.obs[args.target_colname]    
    
    print("X_test shape:", X_test.shape)
    print("Y_test shape:", Y_test.shape)
    print("about to start training or loading models")
    # this is for 1 v all trees
    if args.model_type == "one-vs-all":
        unique_labels = np.unique(Y_train)        
        mask_valid = Y_test.isin(unique_labels)
        if not mask_valid.all():
            removed = set(Y_test[~mask_valid].unique())
            print(f"Warning: removing {len(removed)} unseen test labels: {removed}")

        X_test_filtered = X_test[mask_valid.values]
        Y_test_filtered = Y_test[mask_valid]
        
        all_probs = np.empty((len(Y_test_filtered), len(unique_labels)))

        for i, region_name in enumerate(unique_labels):
            if not model_exists(region_name, output_dir):
                print(f"Starting Random Forest classifier on region {region_name}")
                
                # Create binary labels: 1 for current region, 0 otherwise
                y_train_binary = (Y_train == region_name).astype(int)
                best_params, best_model = train_rf(X_train, y_train_binary, args.search_type)

                # Save the best parameters and model for the current region
                save_model(region_name, output_dir, best_params, best_model)
            else:
                print(f"Only doing eval for {region_name}")
                best_params, best_model = load_model(region_name, output_dir)
            
            # Predict the labels for the test set using the best model
            y_probs = best_model.predict_proba(X_test_filtered)
            assert np.allclose(y_probs.sum(axis=1), 1.0, atol=1e-6), "Not all rows sum to 1"
            all_probs[:, i] = y_probs[:, 1]  # Store probabilities for the positive class
            evaluate_multiclass_and_save(Y_test_filtered, all_probs, unique_labels, output_dir)
    elif args.model_type == "multiclass":
        if not model_exists("multiclass_tree", output_dir):
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
        
        all_probs = best_model.predict_proba(X_test_filtered)
        unique_labels = best_model.classes_
        print("shape of probs and targets is:", all_probs.shape, Y_test_filtered.shape)
        evaluate_multiclass_and_save(Y_test_filtered, all_probs, unique_labels, output_dir)
    elif args.model_type == "regression":
        if not model_exists("regression_tree", output_dir):
            print(f"Starting Random Forest regressor")
            best_params, best_model = train_rf(X_train, Y_train, args.search_type, regression=True)
            save_model("regression_tree", output_dir, best_params, best_model)
        else:
            print(f"Only doing eval for regression")
            best_params, best_model = load_model("regression_tree", output_dir)
        
        all_probs = best_model.predict(X_test).squeeze()
        print(all_probs.shape, Y_test.shape)
        evaluate_regression_and_save(Y_test, all_probs, output_dir)
    else:
        raise ValueError("model_type must be 'multiclass', 'one-vs-all', or 'regression'")

        
if __name__ == "__main__":
    print("Starting random forest experiment script")
    main()
