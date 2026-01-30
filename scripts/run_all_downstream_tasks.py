import argparse
import json
import os
import time
import joblib

import numpy as np
import anndata as ad

from typing import List, Dict, Tuple
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint, uniform
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from trainers.test_functions import evaluate_multiclass_and_save, evaluate_regression_and_save
try:
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    _TABPFN_AVAILABLE = True
except ImportError:
    _TABPFN_AVAILABLE = False

from scripts.tree_learn import train_rf, train_xgb, train_xgb_optuna_native, train_mlp, make_mlp_predictions, load_mlp_model, save_mlp_model, mlp_model_exists, train_linear, save_model, load_model, model_exists

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


def run_random_forest(method_conf: Dict, X_train, Y_train, X_test, Y_test, output_dir, sample_weights=None, run_ova=False):
    # run one vs all + multiclass
    if method_conf["task_type"] == "classification":
        if run_ova:
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
                    best_params, best_model = train_rf(X_train, y_train_binary, method_conf["search_type"], sample_weights)
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
            best_params, best_model = train_rf(X_train, Y_train, method_conf["search_type"], sample_weights)
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
    elif method_conf["task_type"] == "regression":
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


def run_linear(method_conf, X_train, Y_train, X_test, Y_test, output_dir, sample_weights=None, run_ova=False):
    # run one vs all + multiclass
    if method_conf["task_type"] == "classification":
        if run_ova:
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
            print(f"\t [Multiclass Logistic Regression] Starting Random Forest classifier on all regions")
            best_params, best_model = train_linear(X_train, Y_train, method_conf["search_type"])
            save_model("multiclass_lr", multiclass_output_dir, best_params, best_model)
        else:
            print(f"\t [Multiclass Logistic Regression] Only doing eval for all regions")
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
    elif method_conf["task_type"] == "regression":
        print("Running Linear Regression")
        scaler_y = StandardScaler()
        Y_train = scaler_y.fit_transform(Y_train.values.reshape(-1, 1)).squeeze()
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
        all_probs = scaler_y.inverse_transform(all_probs.reshape(-1, 1)).squeeze()
        # print(all_probs.shape, Y_test.shape)
        evaluate_regression_and_save(Y_test, all_probs, reg_output_dir)


def run_xgboost(method_conf, X_train, Y_train, X_test, Y_test, output_dir, sample_weights=None, run_ova=False):
    # run one vs all + multiclass
    if method_conf["task_type"] == "classification":
        if run_ova:
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
                    best_params, best_model = train_xgb(X_train, y_train_binary, method_conf["search_type"])
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
            # best_params, best_model = train_xgb(X_train, Y_train_encoded, method_conf["search_type"], sample_weights)
            best_params, best_model = train_xgb_optuna_native(X_train, Y_train_encoded,  sample_weights=sample_weights, n_trials=150)
            save_model("multiclass_xgboost", multiclass_output_dir, best_params, best_model)
        else:
            print(f"\t [One-vs-All XGBoost] Only doing eval for all regions")
            best_params, best_model = load_model("multiclass_xgboost", multiclass_output_dir)

        all_probs = best_model.predict_proba(X_test)
        unique_labels = best_model.classes_ #le.classes_
        # unique_labels = np.unique(Y_train)
        print("\t shape of probs and targets is:", all_probs.shape, Y_test_encoded.shape)
        evaluate_multiclass_and_save(Y_test_encoded, all_probs, unique_labels, multiclass_output_dir)
    # run regression
    if method_conf["task_type"] == "regression":
        print("Running XGBoost Regression")
        reg_output_dir = f"{output_dir}/regression"
        os.makedirs(reg_output_dir, exist_ok=True)
        if not model_exists("regression_xgboost", reg_output_dir):
            print(f"\t [XGBoost Regression] Starting XGBoost regressor")
            best_params, best_model = train_xgb_optuna_native(X_train, Y_train, regression=True, sample_weights=sample_weights, n_trials=150)
            # best_params, best_model = train_xgb(X_train, Y_train, method_conf["search_type"], sample_weights, regression=True)
            save_model("regression_xgboost", reg_output_dir, best_params, best_model)
        else:
            print(f"\t [XGBoost Regression] Only doing eval for regression")
            best_params, best_model = load_model("regression_xgboost", reg_output_dir)

        all_probs = best_model.predict(X_test).squeeze()
        # print(all_probs.shape, Y_test.shape)
        evaluate_regression_and_save(Y_test, all_probs, reg_output_dir)
        
def run_mlp(method_conf, X_train, Y_train, X_test, Y_test, output_dir, sample_weights=None):
    if method_conf["task_type"] == "classification":
        # multiclass
        print("Running MLP, multiclass")
        multiclass_output_dir = f"{output_dir}/multiclass"
        os.makedirs(multiclass_output_dir, exist_ok=True)
        le = LabelEncoder()
        Y_train_encoded = le.fit_transform(Y_train)
        Y_test_encoded = le.transform(Y_test)
        if not mlp_model_exists("multiclass_mlp", multiclass_output_dir):
            print(f"\t [MULTICLASS MLP] Starting MLP classifier on all regions")
            model = train_mlp(X_train, Y_train_encoded,  class_weights=sample_weights)
            save_mlp_model("multiclass_mlp", multiclass_output_dir, model)
        else:
            print(f"\t [MULTICLASS MLP] Only doing eval for all regions")
            model = load_mlp_model("multiclass_mlp", multiclass_output_dir)

        all_probs = make_mlp_predictions(model, X_test)
        unique_labels = np.unique(Y_train)
        print("\t shape of probs and targets is:", all_probs.shape, Y_test_encoded.shape)
        evaluate_multiclass_and_save(Y_test_encoded, all_probs, unique_labels, multiclass_output_dir)
    # run regression
    if method_conf["task_type"] == "regression":
        print("Running MLP Regression")
        reg_output_dir = f"{output_dir}/regression"
        os.makedirs(reg_output_dir, exist_ok=True)
        if not mlp_model_exists("regression_mlp", reg_output_dir):
            print(f"\t [MLP Regression] Starting MLP regressor")
            model = train_mlp(X_train, Y_train, regression=True, class_weights=sample_weights)
            save_mlp_model("regression_mlp", reg_output_dir, model)
        else:
            print(f"\t [MLP Regression] Only doing eval for regression")
            model = load_mlp_model("regression_mlp", reg_output_dir)

        all_probs = make_mlp_predictions(model, X_test)
        # print(all_probs.shape, Y_test.shape)
        evaluate_regression_and_save(Y_test, all_probs, reg_output_dir)    


def run_tabpfn(method_conf, X_train, Y_train, X_test, Y_test, output_dir):
    if method_conf["task_type"] == "classification":
        # multiclass
        print("Running TabPFN, classification")
        multiclass_output_dir = f"{output_dir}/multiclass"
        os.makedirs(multiclass_output_dir, exist_ok=True)
        clf = TabPFNClassifier(ignore_pretraining_limits=True)
        clf.fit(X_train, Y_train)
        all_probs = clf.predict_proba(X_test)
        unique_labels = np.unique(Y_train)
        print("\t shape of probs and targets is:", all_probs.shape, Y_test.shape)
        evaluate_multiclass_and_save(Y_test, all_probs, unique_labels, multiclass_output_dir)
    # run regression
    if method_conf["task_type"] == "regression":
        print("Running TabPFN Regression")
        reg_output_dir = f"{output_dir}/regression"
        os.makedirs(reg_output_dir, exist_ok=True)
        regr = TabPFNRegressor(ignore_pretraining_limits=True)
        regr.fit(X_train, Y_train)
        all_probs = regr.predict(X_test)
        # print(all_probs.shape, Y_test.shape)
        print("\t shape of probs and targets is:", all_probs.shape, Y_test.shape)
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
            if method_conf["task_type"] == "classification":
                classes = np.unique(Y_train)
                class_weights = compute_class_weight(
                    class_weight="balanced",
                    classes=classes,
                    y=Y_train
                )
                class_weights_dict = dict(zip(classes, class_weights))
                print("Class weights:", list(zip(classes, class_weights)))
                sample_weights = np.array([class_weights_dict[label] for label in Y_train])
            else:
                sample_weights = None
                
            # assert method is implemented
            if method_conf["task_type"] == "classification":
                classes = np.unique(Y_train)
                class_weights = compute_class_weight(
                    class_weight="balanced",
                    classes=classes,
                    y=Y_train
                )
                class_weights_dict = dict(zip(classes, class_weights))
                print("Class weights:", list(zip(classes, class_weights)))
                sample_weights = np.array([class_weights_dict[label] for label in Y_train])
            else:
                sample_weights = None

            if method_name not in ["random_forest", "linear", "xgboost", "tabpfn", "mlp"]:
                print(f"Task '{task_name}' method '{method_name}' is not implemented.")
                continue
            if method_name == "random_forest":
                rf_output_path = f"{embed_output_path}/random_forest"
                os.makedirs(rf_output_path, exist_ok=True)
                print(f"Running random forest for task {task_name}\n")
                run_random_forest(method_conf, X_train, Y_train, X_test, Y_test, rf_output_path, sample_weights=sample_weights)
            elif method_name == "linear":
                linear_output_path = f"{embed_output_path}/linear"
                os.makedirs(linear_output_path, exist_ok=True)
                print(f"Running linear models for task {task_name}\n")
                run_linear(method_conf, X_train, Y_train, X_test, Y_test, linear_output_path, sample_weights=sample_weights)
            elif method_name == "xgboost":
                xgboost_output_path = f"{embed_output_path}/xgboost"
                os.makedirs(xgboost_output_path, exist_ok=True)
                print(f"Running XGBoost models for task {task_name}\n")
                run_xgboost(method_conf, X_train, Y_train, X_test, Y_test, xgboost_output_path, sample_weights=sample_weights)
            elif method_name == "tabpfn":
                if not _TABPFN_AVAILABLE:
                    raise ImportError("TabPFN is not installed.")
                tabpfn_output_path = f"{embed_output_path}/tabpfn"
                os.makedirs(tabpfn_output_path, exist_ok=True)
                print(f"Running TabPFN models for task {task_name}\n")
                run_tabpfn(method_conf, X_train, Y_train, X_test, Y_test, tabpfn_output_path)
            elif method_name == "mlp":
                mlp_output_path = f"{embed_output_path}/mlp"
                os.makedirs(mlp_output_path, exist_ok=True)
                print(f"Running MLP model for task {task_name}\n")
                run_mlp(method_conf, X_train, Y_train, X_test, Y_test, mlp_output_path, sample_weights=sample_weights)
                

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