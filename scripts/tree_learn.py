import numpy as np
import time
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
import xgboost as xgb
import optuna
import json
import joblib
from scipy.stats import randint, uniform
import anndata as ad
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

import argparse
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from trainers.test_functions import evaluate_multiclass_and_save, evaluate_regression_and_save

def train_rf(X_train, y_train, search_type, sample_weights=None, regression=False):
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    
    if regression:
        assert sample_weights is None, "Sample weights not supported for regression in RandomForestRegressor"
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
        rf_model.fit(X_train, y_train, sample_weight=sample_weights)
        return params, rf_model
    
    else:
        raise ValueError("search_type must be 'grid', 'random', or 'none'")

    # Train the model using search
    start_time = time.time()
    search.fit(X_train, y_train, sample_weight=sample_weights)
    end_time = time.time()
    print(f"Time elapsed for search: {(end_time - start_time):.2f} seconds")

    # Get the best parameters and model
    return search.best_params_, search.best_estimator_


def train_xgb(X_train, y_train, search_type, sample_weights=None, regression=False):
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    
    if regression:
        xgb_model = XGBRegressor(
            random_state=42,
            n_jobs=-1,
            tree_method="hist",    # faster training
            eval_metric="rmse"
        )
        search_scoring = 'neg_mean_squared_error'
    else:
        n_classes = len(np.unique(y_train))
        xgb_model = XGBClassifier(
            random_state=42,
            n_jobs=-1,
            tree_method="hist",    # efficient on large datasets
            eval_metric="auc" if n_classes == 2 else "mlogloss"
        )
        search_scoring = 'roc_auc' if n_classes == 2 else 'f1_weighted'

    if search_type == "grid":
        param_grid = {
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 6, 10],
            "n_estimators": [100, 300, 500],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0]
        }
        
        search = GridSearchCV(
            estimator=xgb_model,
            param_grid=param_grid,
            scoring=search_scoring,
            cv=3,
            n_jobs=-1,
            verbose=1
        )
    
    elif search_type == "random":
        param_distributions = {
            "learning_rate": uniform(0.01, 0.3),     # [0.01, 0.31]
            "max_depth": randint(3, 9),
            "min_child_weight": randint(1, 20),
            "n_estimators": randint(100, 1000),
            "subsample": uniform(0.6, 0.3),          # [0.6, 0.9]
            "colsample_bytree": uniform(0.6, 0.3),    # [0.6, 0.9]
            "gamma": uniform(0.0, 5.0)
        }
        
        search = RandomizedSearchCV(
            estimator=xgb_model,
            param_distributions=param_distributions,
            n_iter=70,
            scoring=search_scoring,
            cv=3,
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
    
    elif search_type == "none":
        # params = {
        #     "learning_rate": 0.3,
        #     "max_depth": 10,
        #     "n_estimators": 1500,
        #     "subsample": 0.9,
        #     "colsample_bytree": 0.9,
        # }
        params = {
            "learning_rate": 0.1,
            "max_depth": 3,
            "n_estimators": 50,
            "subsample": 0.6,
            "colsample_bytree": 0.6,
            "gamma": 1.0,
            "min_child_weight": 5
        }
        print("Not doing any search, using fixed parameters:", params)
        xgb_model.set_params(**params)
        xgb_model.fit(X_train, y_train, sample_weight=sample_weights)
        return params, xgb_model
    
    else:
        raise ValueError("search_type must be 'grid', 'random', or 'none'")

    # Run search
    start_time = time.time()
    search.fit(X_train, y_train, sample_weight=sample_weights)
    end_time = time.time()
    print(f"Time elapsed for search: {(end_time - start_time):.2f} seconds")

    return search.best_params_, search.best_estimator_


def train_xgb_optuna_native(
    X_train,
    y_train,
    search_type="optuna",          # keep a similar signature; "none" still supported
    sample_weights=None,
    regression: bool = False,
    n_trials: int = 50,
    cv: int = 3,
    seed: int = 42,
    num_boost_round: int = 5000,    # upper bound; early stopping picks best
    early_stopping_rounds: int = 50,
    timeout: int | None = None,     # seconds; optional
    verbose: bool = True,
):
    """
    Native XGBoost + Optuna hyperparameter tuning.

    Returns:
        best_params_user (dict): params excluding the always-on base params
        final_model (xgboost.Booster): trained booster
        best_num_boost_round (int): chosen number of boosting rounds from CV
        best_cv_score (float): best CV metric (auc higher is better; rmse/mlogloss lower is better)
        study (optuna.study.Study): for inspection / plots
    """
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)

    if verbose:
        print("X_train shape:", X_train.shape)
        print("y_train shape:", y_train.shape)

    # ----------------------------
    # Task setup: objective/metric
    # ----------------------------
    if regression:
        objective = "reg:squarederror"
        eval_metric = "rmse"
        maximize = False
        stratified = False
        extra = {}
        sklearn_model_cls = XGBRegressor
    else:
        classes = np.unique(y_train)
        n_classes = len(classes)
        stratified = True  
        sklearn_model_cls = XGBClassifier

        if n_classes == 2:
            objective = "binary:logistic"
            eval_metric = "auc"
            maximize = True
            extra = {}
        else:
            objective = "multi:softprob"
            # Native CV supports "mlogloss" and "merror" well.
            eval_metric = "mlogloss"
            maximize = False
            extra = {"num_class": int(n_classes)}

    # Always-on params (not tuned)
    base_params = {
        "objective": objective,
        "eval_metric": eval_metric,
        "tree_method": "hist",
        "seed": seed,
        "nthread": -1,
        **extra,
    }

    # DMatrix supports weights natively
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights)

    # --------------------------------
    # "none" mode: train with fixed params
    # --------------------------------
    if search_type == "none":
        user_params = {
            "learning_rate": 0.1,
            "max_depth": 3,
            "subsample": 0.6,
            "colsample_bytree": 0.6,
            "gamma": 1.0,
            "min_child_weight": 5,
            "reg_lambda": 1.0,
            "reg_alpha": 0.0,
        }
        params = {**base_params, **user_params}

        start = time.time()
        cv_res = xgb.cv(
            params=params,
            dtrain=dtrain,
            nfold=cv,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            stratified=stratified,
            seed=seed,
            verbose_eval=verbose,
        )
        best_num_boost_round = len(cv_res)
        metric_col = f"test-{eval_metric}-mean"
        best_cv_score = float(cv_res[metric_col].iloc[-1])

        final_model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=best_num_boost_round,
        )

        if verbose:
            print(f"Time elapsed: {(time.time()-start):.2f} seconds")
            print("Fixed params:", user_params)
            print("Best num_boost_round:", best_num_boost_round)
            print("CV score:", best_cv_score)

        return user_params, final_model, best_num_boost_round, best_cv_score, None

    if search_type not in ("optuna", "random", "grid"):
        raise ValueError("search_type must be 'optuna' or 'none' (or legacy 'random'/'grid').")


    # ----- helper to train final sklearn model -----
    def train_final_sklearn(best_params_user: dict, best_rounds: int):
        # sklearn wrapper uses n_estimators; map best_rounds -> n_estimators
        model = sklearn_model_cls(
            random_state=seed,
            n_jobs=-1,
            tree_method="hist",
            # important: set tuned params
            **best_params_user,
            # important: set n_estimators to best CV round count
            n_estimators=int(best_rounds),
            # set eval_metric for consistent reporting
            eval_metric=eval_metric,
            # ensure multiclass wrapper gets num_class
            **({"num_class": extra["num_class"]} if "num_class" in extra else {}),
        )
        model.fit(X_train, y_train, sample_weight=sample_weights)
        return model

    # ----------------------------
    # Optuna objective: one trial = one param set evaluated by xgb.cv
    # ----------------------------
    def optuna_objective(trial: optuna.trial.Trial) -> float:
        # Core search space (good default for 250–2000 samples)
        user_params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 9),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 30.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
        }

        params = {**base_params, **user_params}

        cv_res = xgb.cv(
            params=params,
            dtrain=dtrain,
            nfold=cv,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            stratified=stratified,
            seed=seed,
            verbose_eval=False,
        )

        best_num_round = len(cv_res)
        metric_col = f"test-{eval_metric}-mean"
        score = float(cv_res[metric_col].iloc[-1])

        # store best round so you can reuse it for final training
        trial.set_user_attr("best_num_boost_round", best_num_round)

        return score  # Optuna will maximize/minimize depending on direction

    direction = "maximize" if maximize else "minimize"
    sampler = optuna.samplers.TPESampler(seed=seed)

    study = optuna.create_study(direction=direction, sampler=sampler)

    start_time = time.time()
    study.optimize(optuna_objective, n_trials=n_trials, timeout=timeout, show_progress_bar=verbose)
    end_time = time.time()

    best_params_user = dict(study.best_trial.params)
    best_num_boost_round = int(study.best_trial.user_attrs["best_num_boost_round"])
    best_cv_score = float(study.best_value)

    if verbose:
        print(f"Time elapsed for Optuna search: {(end_time - start_time):.2f} seconds")
        print("Best params:", best_params_user)
        print("Best num_boost_round:", best_num_boost_round)
        print(f"Best CV {eval_metric}:", best_cv_score)
        
    best_model = train_final_sklearn(best_params_user, best_num_boost_round)
    return best_params_user, best_model

    # # Train final model on ALL training data using best params and best_num_boost_round
    # final_params = {**base_params, **best_params_user}
    # final_model = xgb.train(
    #     params=final_params,
    #     dtrain=dtrain,
    #     num_boost_round=best_num_boost_round,
    # )

    # return best_params_user, final_model, best_num_boost_round, best_cv_score, study




def train_linear(X_train, y_train, search_type, sample_weights=None, regression=False):
    """
    basically same as train_rf, but for linear models
    """
    print("\t X_train shape:", X_train.shape)
    print("\t y_train shape:", y_train.shape)

    # Create pipeline with scaling (important for Lasso)
    if regression:
        ln_model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', Lasso(random_state=42))
        ])
        param_distributions = {
            'classifier__max_iter': [500, 1000, 2000, 5000],
            'classifier__alpha': np.logspace(-4, 2, 10)
        }
        search_scoring = 'neg_mean_squared_error'
    else:
        n_classes = len(np.unique(y_train))
        ln_model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(penalty='l1', solver='saga', random_state=42))
        ])
        param_distributions = {
            'classifier__max_iter': [500, 1000, 2000, 5000],
            'classifier__C': 1.0 / np.logspace(-4, 2, 10)  # C is inverse of alpha in LogisticRegression
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
            verbose=2
        )
    elif search_type == "none": # for debugging
        if regression:
            params = {"classifier__alpha": 1}
        else:
            params = {"classifier__C": 1}
        print("\t Not doing any search, using fixed parameters:", params)
        ln_model.set_params(**params)
        ln_model.fit(X_train, y_train, classifier__sample_weight=sample_weights)
        return params, ln_model
    else:
        raise ValueError(f"\t Unknown search_type: {search_type}")

    # Fit the search
    start_time = time.time()
    search.fit(X_train, y_train, classifier__sample_weight=sample_weights)
    end_time = time.time()
    print(f"\t Time elapsed for search: {(end_time - start_time):.2f} seconds")

    return search.best_params_, search.best_estimator_

def train_mlp(X_train, Y_train_encoded, regression = False, class_weights = None):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    class MLP(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(MLP, self).__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.relu = nn.ReLU()
            self.layer_norm = nn.LayerNorm(hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.relu2 = nn.ReLU()
            self.layer_norm2 = nn.LayerNorm(hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, output_dim)
            

        def forward(self, x):
            out = self.fc1(x)
            out = self.relu(out)
            out = self.layer_norm(out)  
            out = self.fc2(out)
            out = self.relu2(out)
            out = self.layer_norm2(out)  
            out = self.fc3(out)
            return out

    input_dim = X_train.shape[1]
    hidden_dim = 128
    if regression:
        output_dim = 1
    else:
        output_dim = max(len(class_weights),len(torch.unique(torch.tensor(Y_train_encoded))))
    
    
    if regression:
        criterion = torch.nn.MSELoss()
    else:
        criterion=torch.nn.CrossEntropyLoss(weight=class_weights)
    

    model = MLP(input_dim, hidden_dim, output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train_encoded, dtype=torch.long))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    num_epochs = 10
    for epoch in range(num_epochs):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    return model

def make_mlp_predictions(model, X_test):
    import torch
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(X_test, dtype=torch.float32)
        outputs = model(inputs)
        if outputs.shape[1] == 1:
            return outputs.numpy().squeeze()
        else:
            probs = torch.softmax(outputs, dim=1)
            return probs.numpy()


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

def model_exists(label_name, output_dir):
    label_name = label_name.replace("/", " ")
    label_name = label_name.replace(" ", "_")
    label_dir = os.path.join(output_dir, f"{label_name}")
    return os.path.exists(label_dir) and os.path.exists(os.path.join(label_dir, "best_model.pkl")) and os.path.exists(
        os.path.join(label_dir, "best_params.json"))

def load_model(label_name, output_dir):
    label_name = label_name.replace("/", " ")
    label_name = label_name.replace(" ", "_")
    label_dir = os.path.join(output_dir, f"{label_name}")
    with open(os.path.join(label_dir, "best_params.json"), "r") as f:
        best_params = json.load(f)
    best_model = joblib.load(os.path.join(label_dir, "best_model.pkl"))
    return best_params, best_model

def save_mlp_model(label_name, output_dir, model):
    label_name = label_name.replace("/", " ")
    label_name = label_name.replace(" ", "_")
    label_dir = os.path.join(output_dir, f"{label_name}")
    os.makedirs(label_dir, exist_ok=True)
    # Save model
    joblib.dump(model, os.path.join(label_dir, "mlp_model.pkl"))
    
def mlp_model_exists(label_name, output_dir):
    label_name = label_name.replace("/", " ")
    label_name = label_name.replace(" ", "_")
    label_dir = os.path.join(output_dir, f"{label_name}")
    return os.path.exists(label_dir) and os.path.exists(os.path.join(label_dir, "mlp_model.pkl"))

def load_mlp_model(label_name, output_dir):
    label_name = label_name.replace("/", " ")
    label_name = label_name.replace(" ", "_")
    label_dir = os.path.join(output_dir, f"{label_name}")
    model = joblib.load(os.path.join(label_dir, "mlp_model.pkl"))
    return model
    


def main():
    parser = argparse.ArgumentParser(description="Script for processing embeddings.")

    parser.add_argument("--train-embed-path", type=str, required=True, help="Path to the anndata where the train embeddings are saved or should be saved in the obsm['embedding'].")
    parser.add_argument("--test-embed-path", type=str, required=True, help="Path to the anndata where the test embeddings are saved or should be saved in the obsm['embedding'].")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the best model and parameters.")
    parser.add_argument("--model-arch", type=str, choices=["rf", "xgb"], default="rf", help="Type of model to use: 'rf' for Random Forest, 'xgb' for XGBoost.")
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
    
    if args.model_arch == "rf":
        train_func = train_rf
    else:
        train_func = train_xgb
        
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
    if args.model_type == "one-vs-all" or args.model_type == "multiclass":
        le = LabelEncoder()
        Y_train = le.fit_transform(Y_train)
        
        unique_labels = le.classes_       
        mask_valid = Y_test.isin(unique_labels)
        if not mask_valid.all():
            removed = set(Y_test[~mask_valid].unique())
            print(f"Warning: removing {len(removed)} unseen test labels: {removed}")

        X_test_filtered = X_test[mask_valid.values]
        Y_test_filtered = Y_test[mask_valid]
        Y_test_filtered = le.transform(Y_test_filtered)
        
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(Y_train),
            y=Y_train
        )
        class_weights_dict = dict(zip(np.unique(Y_train), class_weights))
        print("Class weights:", list(zip(le.classes_, class_weights)))
        sample_weights = np.array([class_weights_dict[label] for label in Y_train])
        
        if args.model_type == "one-vs-all":
            raise NotImplementedError("One-vs-all is not implemented in this version.")
        # all_probs = np.empty((len(Y_test_filtered), len(unique_labels)))
        # for i, label_name in enumerate(unique_labels):
        #     if not model_exists(label_name, output_dir):
        #         print(f"Starting classifier on label {label_name}")
                
        #         # Create binary labels: 1 for current label, 0 otherwise
        #         y_train_binary = (Y_train == label_name).astype(int)
        #         # sample_weights = 
        #         # best_params, best_model = train_func(X_train, y_train_binary, args.search_type, sample_weights)

        #         # Save the best parameters and model for the current label
        #         save_model(label_name, output_dir, best_params, best_model)
        #     else:
        #         print(f"Only doing eval for {label_name}")
        #         best_params, best_model = load_model(label_name, output_dir)
            
        #     # Predict the labels for the test set using the best model
        #     y_probs = best_model.predict_proba(X_test_filtered)
        #     assert np.allclose(y_probs.sum(axis=1), 1.0, atol=1e-6), "Not all rows sum to 1"
        #     all_probs[:, i] = y_probs[:, 1]  # Store probabilities for the positive class
        #     evaluate_multiclass_and_save(Y_test_filtered, all_probs, unique_labels, output_dir)
        elif args.model_type == "multiclass":
            if not model_exists("multiclass_tree", output_dir):
                print(f"Starting classifier on all labels")
                best_params, best_model = train_func(X_train, Y_train, args.search_type, sample_weights)
                save_model("multiclass_tree", output_dir, best_params, best_model)
            else:
                print(f"Only doing eval for all labels")
                best_params, best_model = load_model("multiclass_tree", output_dir)
            
            all_probs = best_model.predict_proba(X_test_filtered)
            print("shape of probs and targets is:", all_probs.shape, Y_test_filtered.shape)
            Y_test_filtered = le.inverse_transform(Y_test_filtered)
            evaluate_multiclass_and_save(Y_test_filtered, all_probs, unique_labels, output_dir)
    elif args.model_type == "regression":
        if not model_exists("regression_tree", output_dir):
            print(f"Starting regressor")
            best_params, best_model = train_func(X_train, Y_train, args.search_type, regression=True)
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
    print("Starting training tree experiment script")
    main()
