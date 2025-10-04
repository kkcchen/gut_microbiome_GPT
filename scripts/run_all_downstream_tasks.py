import argparse
import json
import os
import time
import joblib

import numpy as np
import anndata as ad

from typing import List, Dict, Tuple
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint, uniform
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier, XGBRegressor
from trainers.test_functions import evaluate_multiclass_and_save, evaluate_regression_and_save


def tasks_type(string: str) -> Dict:
    """Convert JSON path to a dict with hard format, this is the expected type of task input,
    the format should be
    {<task_name: str>: {
        "label_type": choices=["categorical_label", "continuous_label"],
        "ignored_labels": List[str], default=[],
        "embed_name": List[str], default=["embedding"],
        "target_colname": str,  default="location",
        "methods": {
            "random_forest":{
                "search_type": str, choices=["grid", "random", "none"],
            }
    }
    e.g. {"age": {
            "label_type": "continuous_label", # this is the column name to use in the anndata
            "ignored_labels": [], # if a label is in this list, it is ignored
            "embed_name": ["embedding"], # Name of the obsm key where embeddings are stored.
            "target_colname": "age", # name of the target.
            "methods": {
                "random forest": {
                    "search_type": "grid",
                    },
                "mlp": #TODO: not implemented yet
                },
            }
        }
    """
    try:
        with open(string) as f:
            data = json.load(f)
        f.close()
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"Invalid file: {e}")

    if not isinstance(data, dict):
        raise argparse.ArgumentTypeError("Top level must be a dictionary mapping task names to task configs.")

    for task, conf in data.items():
        if not isinstance(conf, dict):
            raise argparse.ArgumentTypeError(f"Config for task '{task}' must be a dictionary.")

        # label_type
        if "label_type" not in conf or conf["label_type"] not in ["categorical_label", "continuous_label"]:
            raise argparse.ArgumentTypeError(
                f"Task '{task}': 'label_type' must be one of ['categorical_label', 'continuous_label'].")

        # ignored_labels
        if "ignored_labels" in conf and not isinstance(conf["ignored_labels"], list):
            raise argparse.ArgumentTypeError(f"Task '{task}': 'ignored_labels' should be a list.")
        conf.setdefault("ignored_labels", [])

        # methods
        if "methods" not in conf or not isinstance(conf["methods"], dict):
            raise argparse.ArgumentTypeError(f"Task '{task}': 'methods' must be a dictionary.")

        for method_name, method_conf in conf["methods"].items():
            if not isinstance(method_conf, dict):
                raise argparse.ArgumentTypeError(f"Task '{task}' method '{method_name}': config must be a dictionary.")
            # validate random forest
            if method_name == "random_forest":
                # search_type
                if ("search_type" not in method_conf
                        or method_conf["search_type"] not in ["grid", "random", "none"]):
                    raise argparse.ArgumentTypeError(
                        f"Task '{task}' random_forest: 'search_type' must be one of ['grid', 'random', 'none']."
                    )
                # model_types
                mt = method_conf.get("model_types", [])
    return data


def train_rf(X_train, y_train, search_type, regression=False):
    """
    from Kevin Chen, original code for training random forest
    """
    print("\t X_train shape:", X_train.shape)
    print("\t y_train shape:", y_train.shape)

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
            "max_samples": [1.0, 0.75, 0.5],
            "max_features": [0.1, 0.2, 0.3],
            "n_estimators": [50, 250, 500, 1000]
        }

        # Perform grid search with cross-validation
        search = GridSearchCV(
            estimator=rf_model,
            param_grid=param_grid,
            scoring=search_scoring,
            cv=3,
            n_jobs=-1,
            verbose=1
        )
    elif search_type == "random":
        param_distributions = {  # discrete for min_samples_leaf?
            "min_samples_leaf": randint(2, 20),
            "max_samples": uniform(0.5, 0.5),
            "max_features": uniform(0.5, 0.5),
            "n_estimators": randint(200, 1000)
        }

        # Perform random search with cross-validation
        search = RandomizedSearchCV(
            estimator=rf_model,
            param_distributions=param_distributions,
            n_iter=10,  # Number of parameter combinations to try
            scoring=search_scoring,
            cv=3,
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
    elif search_type == "none":
        params = {"max_features": 0.3446384285364502, "max_samples": 0.8534286719238086, "min_samples_leaf": 3,
                  "n_estimators": 10}
        print("\t Not doing any search, using fixed parameters:", params)
        rf_model.set_params(**params)
        rf_model.fit(X_train, y_train)
        return params, rf_model
    else:
        raise ValueError("search_type must be 'grid', 'random', or 'none'")

    # Train the model using search
    start_time = time.time()
    search.fit(X_train, y_train)
    end_time = time.time()
    print(f"\t Time elapsed for search: {(end_time - start_time):.2f} seconds")

    # Get the best parameters and model
    return search.best_params_, search.best_estimator_

def train_linear(X_train, y_train, search_type, regression=False):
    """
    basically same as train_rf, but for linear models
    """
    print("\t X_train shape:", X_train.shape)
    print("\t y_train shape:", y_train.shape)

    # Create pipeline with scaling (important for Lasso)
    if regression:
        ln_model = Pipeline([
            ('scaler', StandardScaler()),
            ('lasso', Lasso(random_state=42))
        ])
        param_distributions = {
            'lasso__max_iter': [500, 1000, 2000, 5000],
            'lasso__alpha': np.logspace(-4, 2, 10)
        }
        search_scoring = 'neg_mean_squared_error'
    else:
        n_classes = len(np.unique(y_train))
        ln_model = Pipeline([
            ('scaler', StandardScaler()),
            ('logistic', LogisticRegression(penalty='l1', solver='liblinear', random_state=42))
        ])
        param_distributions = {
            'logistic__max_iter': [500, 1000, 2000, 5000],
            'logistic__C': 1.0 / np.logspace(-4, 2, 10)  # C is inverse of alpha in LogisticRegression
        }
        search_scoring = 'roc_auc' if n_classes == 2 else 'f1_weighted'

    # hyperparameter search
    if search_type == "grid":
        search = GridSearchCV(
            ln_model,
            param_distributions,
            cv=5,
            scoring=search_scoring,
            n_jobs=-1,
            verbose=1
        )
    elif search_type == "random":
        search = RandomizedSearchCV(
            ln_model,
            param_distributions,
            cv=5,
            scoring=search_scoring,
            n_jobs=-1,
            n_iter=5,
            random_state=42,
            verbose=1
        )
    elif search_type == "none": # for debugging
        if regression:
            params = {"lasso__alpha": 1}
        else:
            params = {"logistic__C": 1}
        print("\t Not doing any search, using fixed parameters:", params)
        ln_model.set_params(**params)
        ln_model.fit(X_train, y_train)
        return params, ln_model
    else:
        raise ValueError(f"\t Unknown search_type: {search_type}")

    # Fit the search
    start_time = time.time()
    search.fit(X_train, y_train)
    end_time = time.time()
    print(f"\t Time elapsed for search: {(end_time - start_time):.2f} seconds")

    return search.best_params_, search.best_estimator_


def train_xgboost(X_train, y_train, search_type, regression=False):
    """
    basically same as train_rf, but for xgboost
    """
    print("\t X_train shape:", X_train.shape)
    print("\t y_train shape:", y_train.shape)

    if regression:
        xgb_model = Pipeline([
            ('scaler', StandardScaler()),
            ('xgb', XGBRegressor(tree_method='hist', random_state=42))
        ])
        param_distributions = {
            'xgb__n_estimators': [100, 200],
            'xgb__max_depth': [3, 5],
            'xgb__learning_rate': [0.01, 0.1],
            'xgb__subsample': [0.8, 1.0]
        }
        search_scoring = 'neg_mean_squared_error'
    else:
        xgb_model = Pipeline([
            ('scaler', StandardScaler()),
            ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', tree_method='hist', random_state=42))
        ])
        param_distributions = {
            'xgb__n_estimators': [100, 200, 300],
            'xgb__max_depth': [3, 6, 10],
            'xgb__learning_rate': [0.01, 0.1, 0.3],
            'xgb__subsample': [0.7, 0.8, 1.0]
        }
        search_scoring = 'accuracy'

    # hyperparameter search
    if search_type == "grid":
        search = GridSearchCV(
            xgb_model,
            param_distributions,
            cv=5,
            scoring=search_scoring,
            n_jobs=-1,
            verbose=1
        )
    elif search_type == "random":
        search = RandomizedSearchCV(
            xgb_model,
            param_distributions,
            cv=5,
            scoring=search_scoring,
            n_jobs=-1,
            n_iter=5,
            random_state=42,
            verbose=1
        )
    elif search_type == "none": # for debugging
        params = {
            'xgb__n_estimators': 10,
            'xgb__max_depth': 3,
            'xgb__learning_rate': 0.01,
            'xgb__subsample': 1.0
        }
        print("\t Not doing any search, using fixed parameters:", params)
        xgb_model.set_params(**params)
        xgb_model.fit(X_train, y_train)
        return params, xgb_model
    else:
        raise ValueError(f"\t Unknown search_type: {search_type}")

    # Fit the search
    start_time = time.time()
    search.fit(X_train, y_train)
    end_time = time.time()
    print(f"\t Time elapsed for search: {(end_time - start_time):.2f} seconds")

    return search.best_params_, search.best_estimator_


def model_exists(label_name, output_dir):
    label_name = label_name.replace("/", " ")
    label_name = label_name.replace(" ", "_")
    region_dir = os.path.join(output_dir, f"{label_name}")
    return os.path.exists(region_dir) and os.path.exists(os.path.join(region_dir, "best_model.pkl")) and os.path.exists(
        os.path.join(region_dir, "best_params.json"))

def save_model(label_name, output_dir, best_params, best_model):
    label_name = label_name.replace("/", " ")
    label_name = label_name.replace(" ", "_")
    label_dir = os.path.join(output_dir, f"{label_name}")
    os.makedirs(label_dir, exist_ok=True)

    # Save best parameters
    with open(os.path.join(label_dir, "best_params.json"), "w") as f:
        json.dump(best_params, f)

    # Save best model
    joblib.dump(best_model, os.path.join(label_dir, "best_model.pkl"))

def load_model(label_name, output_dir):
    label_name = label_name.replace("/", " ")
    label_name = label_name.replace(" ", "_")
    label_dir = os.path.join(output_dir, f"{label_name}")
    with open(os.path.join(label_dir, "best_params.json"), "r") as f:
        best_params = json.load(f)
    best_model = joblib.load(os.path.join(label_dir, "best_model.pkl"))
    return best_params, best_model

def run_random_forest(method_conf: Dict, X_train, Y_train, X_test, Y_test, output_dir):
    # run one vs all + multiclass
    if method_conf["task_type"] == "classification":
        # one vs all
        print("Running Random Forest Classification, one vs all")
        unique_labels = np.unique(Y_train)
        mask_valid = Y_test.isin(unique_labels)
        if not mask_valid.all():
            raise Exception("this dataset contains unseen labels in test")
        ova_probs = np.empty((len(Y_test), len(unique_labels)))
        ova_output_dir = f"{output_dir}/ova"
        os.makedirs(ova_output_dir, exist_ok=True)

        for i, label_name in enumerate(unique_labels):
            if not model_exists(label_name, ova_output_dir):
                print(f"\t [One-vs-All Random Forest] Starting Random Forest classifier on label {label_name}")
                y_train_binary = (Y_train == label_name).astype(int)
                best_params, best_model = train_rf(X_train, y_train_binary, method_conf["search_type"])
                save_model(label_name, ova_output_dir, best_params, best_model)
            else:
                print(f"\t [One-vs-All Random Forest] Only doing eval for {label_name}")
                best_params, best_model = load_model(label_name, ova_output_dir)

            # Predict the labels for the test set using the best model
            y_probs = best_model.predict_proba(X_test)
            assert np.allclose(y_probs.sum(axis=1), 1.0, atol=1e-6), "\t[One-vs-All Random Forest] Not all rows sum to 1"
            ova_probs[:, i] = y_probs[:, 1]  # Store probabilities for the positive class
        evaluate_multiclass_and_save(Y_test, ova_probs, unique_labels, ova_output_dir)
        # multiclass
        print("Running Random Forest Classification, multiclass")
        multiclass_output_dir = f"{output_dir}/multiclass"
        os.makedirs(multiclass_output_dir, exist_ok=True)
        if not model_exists("multiclass_tree", multiclass_output_dir):
            print(f"\t [Multiclass Random Forest] Starting Random Forest classifier on all regions")
            best_params, best_model = train_rf(X_train, Y_train, method_conf["search_type"])
            save_model("multiclass_tree", multiclass_output_dir, best_params, best_model)
        else:
            print(f"\t [Multiclass Random Forest] Only doing eval for all regions")
            best_params, best_model = load_model("multiclass_tree", multiclass_output_dir)
        # Filter test samples with unseen classes BEFORE prediction
        valid_classes = set(best_model.classes_)
        mask_valid = Y_test.isin(valid_classes)

        if not mask_valid.all():
            removed_labels = set(Y_test[~mask_valid].unique())
            print(f"\t Warning: removing test samples with unseen labels: {removed_labels}")

        X_test_filtered = X_test[mask_valid.values]
        Y_test_filtered = Y_test[mask_valid]

        all_probs = best_model.predict_proba(X_test_filtered)
        unique_labels = best_model.classes_
        print("\t shape of probs and targets is:", all_probs.shape, Y_test_filtered.shape)
        evaluate_multiclass_and_save(Y_test_filtered, all_probs, unique_labels, multiclass_output_dir)
    # run regression
    if method_conf["task_type"] == "regression":
        print("Running Random Forest Regression")
        reg_output_dir = f"{output_dir}/regression"
        os.makedirs(reg_output_dir, exist_ok=True)
        if not model_exists("regression_tree", reg_output_dir):
            print(f"\t [Regression Random Forest] Starting Random Forest regressor")
            best_params, best_model = train_rf(X_train, Y_train, method_conf["search_type"], regression=True)
            save_model("regression_tree", reg_output_dir, best_params, best_model)
        else:
            print(f"\t [Regression Random Forest] Only doing eval for regression")
            best_params, best_model = load_model("regression_tree", reg_output_dir)

        all_probs = best_model.predict(X_test).squeeze()
        # print(all_probs.shape, Y_test.shape)
        evaluate_regression_and_save(Y_test, all_probs, reg_output_dir)


def run_linear(method_conf, X_train, Y_train, X_test, Y_test, output_dir):
    # run one vs all + multiclass
    if method_conf["task_type"] == "classification":
        # one vs all
        print("Running Logistic Regression, one vs all")
        unique_labels = np.unique(Y_train)
        mask_valid = Y_test.isin(unique_labels)
        if not mask_valid.all():
            raise Exception("this dataset contains unseen labels in test")
        ova_probs = np.empty((len(Y_test), len(unique_labels)))
        ova_output_dir = f"{output_dir}/ova"
        os.makedirs(ova_output_dir, exist_ok=True)

        for i, label_name in enumerate(unique_labels):
            if not model_exists(label_name, ova_output_dir):
                print(f"\t [One-vs-All Logistic Regression] Starting logistic classifier on label {label_name}")
                y_train_binary = (Y_train == label_name).astype(int)
                best_params, best_model = train_linear(X_train, y_train_binary, method_conf["search_type"])
                save_model(label_name, ova_output_dir, best_params, best_model)
            else:
                print(f"\t [One-vs-All Logistic Regression] Only doing eval for {label_name}")
                best_params, best_model = load_model(label_name, ova_output_dir)

            # Predict the labels for the test set using the best model
            y_probs = best_model.predict_proba(X_test)
            assert np.allclose(y_probs.sum(axis=1), 1.0,
                               atol=1e-6), "\t[One-vs-All Logistic Regression] Not all rows sum to 1"
            ova_probs[:, i] = y_probs[:, 1]  # Store probabilities for the positive class
        evaluate_multiclass_and_save(Y_test, ova_probs, unique_labels, ova_output_dir)
        # multiclass
        print("Running Logistic Regression, multiclass")
        multiclass_output_dir = f"{output_dir}/multiclass"
        os.makedirs(multiclass_output_dir, exist_ok=True)
        if not model_exists("multiclass_lr", multiclass_output_dir):
            print(f"\t [One-vs-All Logistic Regression] Starting Random Forest classifier on all regions")
            best_params, best_model = train_linear(X_train, Y_train, method_conf["search_type"])
            save_model("multiclass_lr", multiclass_output_dir, best_params, best_model)
        else:
            print(f"\t [One-vs-All Logistic Regression] Only doing eval for all regions")
            best_params, best_model = load_model("multiclass_lr", multiclass_output_dir)
        # Filter test samples with unseen classes BEFORE prediction
        valid_classes = set(best_model.classes_)
        mask_valid = Y_test.isin(valid_classes)

        if not mask_valid.all():
            removed_labels = set(Y_test[~mask_valid].unique())
            print(f"\t Warning: removing test samples with unseen labels: {removed_labels}")

        X_test_filtered = X_test[mask_valid.values]
        Y_test_filtered = Y_test[mask_valid]

        all_probs = best_model.predict_proba(X_test_filtered)
        unique_labels = best_model.classes_
        print("\t shape of probs and targets is:", all_probs.shape, Y_test_filtered.shape)
        evaluate_multiclass_and_save(Y_test_filtered, all_probs, unique_labels, multiclass_output_dir)
    # run regression
    if method_conf["task_type"] == "regression":
        print("Running Linear Regression")
        reg_output_dir = f"{output_dir}/regression"
        os.makedirs(reg_output_dir, exist_ok=True)
        if not model_exists("lasso_regression", reg_output_dir):
            print(f"\t [Linear Regression with Lasso] Starting Linear regressor with Lasso")
            best_params, best_model = train_linear(X_train, Y_train, method_conf["search_type"], regression=True)
            save_model("lasso_regression", reg_output_dir, best_params, best_model)
        else:
            print(f"\t [Linear Regression with Lasso] Only doing eval for regression")
            best_params, best_model = load_model("lasso_regression", reg_output_dir)

        all_probs = best_model.predict(X_test).squeeze()
        # print(all_probs.shape, Y_test.shape)
        evaluate_regression_and_save(Y_test, all_probs, reg_output_dir)


def run_xgboost(method_conf, X_train, Y_train, X_test, Y_test, output_dir):
    # run one vs all + multiclass
    if method_conf["task_type"] == "classification":
        # one vs all
        print("Running XGBoost, one vs all")
        unique_labels = np.unique(Y_train)
        mask_valid = Y_test.isin(unique_labels)
        if not mask_valid.all():
            raise Exception("this dataset contains unseen labels in test")
        ova_probs = np.empty((len(Y_test), len(unique_labels)))
        ova_output_dir = f"{output_dir}/ova"
        os.makedirs(ova_output_dir, exist_ok=True)

        for i, label_name in enumerate(unique_labels):
            if not model_exists(label_name, ova_output_dir):
                print(f"\t [One-vs-All XGBoost] Starting xgboost classifier on label {label_name}")
                y_train_binary = (Y_train == label_name).astype(int)
                best_params, best_model = train_xgboost(X_train, y_train_binary, method_conf["search_type"])
                save_model(label_name, ova_output_dir, best_params, best_model)
            else:
                print(f"\t [One-vs-All XGBoost] Only doing eval for {label_name}")
                best_params, best_model = load_model(label_name, ova_output_dir)

            # Predict the labels for the test set using the best model
            y_probs = best_model.predict_proba(X_test)
            assert np.allclose(y_probs.sum(axis=1), 1.0,
                               atol=1e-6), "\t[One-vs-All XGBoost] Not all rows sum to 1"
            ova_probs[:, i] = y_probs[:, 1]  # Store probabilities for the positive class
        evaluate_multiclass_and_save(Y_test, ova_probs, unique_labels, ova_output_dir)
        # multiclass
        print("Running XGBoost, multiclass")
        multiclass_output_dir = f"{output_dir}/multiclass"
        os.makedirs(multiclass_output_dir, exist_ok=True)
        le = LabelEncoder()
        Y_train_encoded = le.fit_transform(Y_train)
        Y_test_encoded = le.transform(Y_test)
        if not model_exists("multiclass_xgboost", multiclass_output_dir):
            print(f"\t [One-vs-All XGBoost] Starting XGBoost classifier on all regions")
            best_params, best_model = train_xgboost(X_train, Y_train_encoded, method_conf["search_type"])
            save_model("multiclass_xgboost", multiclass_output_dir, best_params, best_model)
        else:
            print(f"\t [One-vs-All XGBoost] Only doing eval for all regions")
            best_params, best_model = load_model("multiclass_xgboost", multiclass_output_dir)

        all_probs = best_model.predict_proba(X_test)
        unique_labels = le.classes_
        # unique_labels = np.unique(Y_train)
        print("\t shape of probs and targets is:", all_probs.shape, Y_test_encoded.shape)
        evaluate_multiclass_and_save(Y_test, all_probs, unique_labels, multiclass_output_dir)
    # run regression
    if method_conf["task_type"] == "regression":
        print("Running XGBoost Regression")
        reg_output_dir = f"{output_dir}/regression"
        os.makedirs(reg_output_dir, exist_ok=True)
        if not model_exists("regression_xgboost", reg_output_dir):
            print(f"\t [XGBoost Regression] Starting XGBoost regressor")
            best_params, best_model = train_xgboost(X_train, Y_train, method_conf["search_type"], regression=True)
            save_model("regression_xgboost", reg_output_dir, best_params, best_model)
        else:
            print(f"\t [XGBoost Regression] Only doing eval for regression")
            best_params, best_model = load_model("regression_xgboost", reg_output_dir)

        all_probs = best_model.predict(X_test).squeeze()
        # print(all_probs.shape, Y_test.shape)
        evaluate_regression_and_save(Y_test, all_probs, reg_output_dir)


def run_task(task_name: str,
             task_config: Dict,
             adata_train_all: ad.AnnData,
             adata_test_all: ad.AnnData,
             output_path: str):
    methods = task_config["methods"]
    # handle data
    adata_train_task = adata_train_all[adata_train_all.obs[task_config["target_colname"]] == task_name]
    adata_train_task = adata_train_task[~adata_train_task.obs[task_config["label_type"]].isin(task_config["ignored_labels"])]
    adata_test_task = adata_test_all[adata_test_all.obs[task_config["target_colname"]] == task_name]
    adata_test_task = adata_test_task[~adata_test_task.obs[task_config["label_type"]].isin(task_config["ignored_labels"])]

    # iterate thru all embeds, e.g. if we are running pipeline on raw and pretrained, we want two sets of outputs
    for embed_name in task_config["embed_name"]:
        X_train = adata_train_task.obsm[embed_name]
        Y_train = adata_train_task.obs[task_config["label_type"]]
        X_test = adata_test_task.obsm[embed_name]
        Y_test = adata_test_task.obs[task_config["label_type"]]
        embed_output_path = f"{output_path}/{embed_name}"
        os.makedirs(embed_output_path, exist_ok=True)
        for method_name, method_conf in methods.items():
            # assert method is implemented
            if method_name not in ["random_forest", "linear", "xgboost"]:
                print(f"Task '{task_name}' method '{method_name}' is not implemented.")
                continue
            if method_name == "random_forest":
                rf_output_path = f"{embed_output_path}/random_forest"
                os.makedirs(rf_output_path, exist_ok=True)
                print(f"Running random forest for task {task_name}\n")
                run_random_forest(method_conf, X_train, Y_train, X_test, Y_test, rf_output_path)
            elif method_name == "linear":
                linear_output_path = f"{embed_output_path}/linear"
                os.makedirs(linear_output_path, exist_ok=True)
                print(f"Running linear models for task {task_name}\n")
                run_linear(method_conf, X_train, Y_train, X_test, Y_test, linear_output_path)
            elif method_name == "xgboost":
                xgboost_output_path = f"{embed_output_path}/xgboost"
                os.makedirs(xgboost_output_path, exist_ok=True)
                print(f"Running XGBoost models for task {task_name}\n")
                run_xgboost(method_conf, X_train, Y_train, X_test, Y_test, xgboost_output_path)

def main():
    parser = argparse.ArgumentParser(description="Script for running all downstream tasks")

    # -------- DATASET PATHS
    parser.add_argument("--train-path", type=str, required=True, help="Path to the training ann data")
    parser.add_argument("--test-path", type=str, required=True, help="Path to the test ann data")
    parser.add_argument("--output-path", type=str, required=True, help="Path to the outputs directory")
    # -------- TASKS AND OBJECTIVES
    parser.add_argument("--tasks", type=str, required=True, help="path to json file for downstream tasks")

    args = parser.parse_args()
    tasks = tasks_type(args.tasks)
    output_path = args.output_path

    # load anndata objects
    adata_train = ad.read_h5ad(args.train_path)
    adata_test = ad.read_h5ad(args.test_path)

    # process tasks
    for task_name, task_config in tasks.items():
        task_output_path = f"{output_path}/{task_name}"
        os.makedirs(task_output_path, exist_ok=True)
        run_task(task_name, task_config, adata_train, adata_test, task_output_path)


if __name__ == "__main__":
    print("Running all downstream tasks")
    main()