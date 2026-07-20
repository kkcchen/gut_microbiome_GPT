"""
Downstream Task Model Training Functions
Pure ML logic - add new methods here without touching orchestration code.
Each function follows the same signature for easy registration.
"""

import time
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Any, Optional
import joblib
import json

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint, uniform, loguniform
from sklearn.preprocessing import LabelEncoder

from trainers import logger

try:
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False


# ==============================================================================
# MODEL TRAINING FUNCTIONS
# Each function returns (params_dict, trained_model)
# ==============================================================================

def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    search_type: str = "none",
    sample_weights: Optional[np.ndarray] = None,
    task_type: str = "classification"
) -> Tuple[Dict, Any]:
    """
    Train Random Forest model.

    Args:
        X_train: Training features
        y_train: Training labels
        search_type: 'grid', 'random', or 'none'
        sample_weights: Optional sample weights
        task_type: 'classification' or 'regression'

    Returns:
        Tuple of (best_params, best_model)
    """
    regression = (task_type == "regression")
    logger.info(f"  Training Random Forest ({'Regression' if regression else 'Classification'})")
    logger.info(f"    X_train: {X_train.shape}, y_train: {y_train.shape}")

    if regression:
        model = RandomForestRegressor(
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )
        search_scoring = 'neg_mean_squared_error'
    else:
        model = RandomForestClassifier(
            bootstrap=True,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        )
        n_classes = len(np.unique(y_train))
        search_scoring = 'roc_auc' if n_classes == 2 else 'f1_weighted'

    if search_type == "grid":
        param_grid = {
            "min_samples_leaf": [1, 10],
            "n_estimators": [250, 500]
        }
        search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring=search_scoring,
            cv=3,
            n_jobs=-1,
            verbose=1
        )
        start_time = time.time()
        search.fit(X_train, y_train, sample_weight=sample_weights)
        logger.info(f"    Search completed in {time.time() - start_time:.2f}s")
        return search.best_params_, search.best_estimator_

    elif search_type == "random":
        param_distributions = {
            "min_samples_leaf": randint(2, 20),
            "max_samples": uniform(0.5, 0.5),
            "max_features": uniform(0.5, 0.5),
            "n_estimators": randint(200, 1000)
        }
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=10,
            scoring=search_scoring,
            cv=3,
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        start_time = time.time()
        search.fit(X_train, y_train, sample_weight=sample_weights)
        logger.info(f"    Search completed in {time.time() - start_time:.2f}s")
        return search.best_params_, search.best_estimator_

    elif search_type == "none":
        params = {
            "max_features": 0.7,
            "max_samples": 0.85,
            "min_samples_leaf": 3,
            "n_estimators": 500
        }
        logger.info(f"    Using fixed parameters: {params}")
        model.set_params(**params)
        model.fit(X_train, y_train, sample_weight=sample_weights)
        return params, model

    else:
        raise ValueError(f"search_type must be 'grid', 'random', or 'none', got: {search_type}")


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    search_type: str = "none",
    sample_weights: Optional[np.ndarray] = None,
    task_type: str = "classification"
) -> Tuple[Dict, Any]:
    """
    Train XGBoost model.

    Args:
        X_train: Training features
        y_train: Training labels
        search_type: 'grid', 'random', or 'none'
        sample_weights: Optional sample weights
        task_type: 'classification' or 'regression'

    Returns:
        Tuple of (best_params, best_model)
    """
    regression = (task_type == "regression")
    logger.info(f"  Training XGBoost ({'Regression' if regression else 'Classification'})")
    logger.info(f"    X_train: {X_train.shape}, y_train: {y_train.shape}")

    if regression:
        model = XGBRegressor(
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
            eval_metric="rmse"
        )
        search_scoring = 'neg_mean_squared_error'
        y_train_encoded = y_train
    else:
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)

        n_classes = len(np.unique(y_train))
        model = XGBClassifier(
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
            use_label_encoder=False,
            eval_metric="auc" if n_classes == 2 else "mlogloss"
        )
        search_scoring = 'roc_auc' if n_classes == 2 else 'roc_auc_ovr_weighted'

    if search_type == "grid":
        param_grid = {
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 6],
            "n_estimators": [200, 500]
        }
        search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring=search_scoring,
            cv=3,
            n_jobs=-1,
            verbose=1
        )
        start_time = time.time()
        search.fit(X_train, y_train_encoded, sample_weight=sample_weights)
        logger.info(f"    Search completed in {time.time() - start_time:.2f}s")
        best_model = search.best_estimator_
        best_params = search.best_params_
    elif search_type == "random":
        param_distributions = {
            "learning_rate": loguniform(1e-3, 3e-1),
            "max_depth": randint(3, 12),
            "n_estimators": randint(100, 1000),
            "subsample": uniform(0.5, 0.5),
            "colsample_bytree": uniform(0.5, 0.5),
            "min_child_weight": loguniform(1e-1, 1e2),
            "gamma": loguniform(1e-8, 1e1),
            "reg_alpha": loguniform(1e-8, 1e1),
            "reg_lambda": loguniform(1e-3, 1e2),
        }
        
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=200,
            scoring=search_scoring,
            cv=5,
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        start_time = time.time()
        search.fit(X_train, y_train_encoded, sample_weight=sample_weights)
        logger.info(f"    Search completed in {time.time() - start_time:.2f}s")
        best_model = search.best_estimator_
        best_params = search.best_params_
    elif search_type == "none":
        params = {
            "learning_rate": 0.05,
            "max_depth": 7,
            "n_estimators": 500,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "tree_method": "hist"
        }
        logger.info(f"    Using fixed parameters: {params}")
        model.set_params(**params)
        model.fit(X_train, y_train_encoded, sample_weight=sample_weights)
        best_model = model
        best_params = params
    else:
        raise ValueError(f"search_type must be 'grid', 'random', or 'none', got: {search_type}")

    if not regression:
        best_model._label_encoder = label_encoder
    
    return best_params, best_model


def train_linear(
    X_train: np.ndarray,
    y_train: np.ndarray,
    search_type: str = "none",
    sample_weights: Optional[np.ndarray] = None,
    task_type: str = "classification"
) -> Tuple[Dict, Any]:
    """
    Train linear model (Logistic Regression or Lasso).

    Args:
        X_train: Training features
        y_train: Training labels
        search_type: 'grid', 'random', or 'none'
        sample_weights: Optional sample weights
        task_type: 'classification' or 'regression'

    Returns:
        Tuple of (best_params, best_model)
    """
    regression = (task_type == "regression")
    logger.info(f"  Training Linear Model ({'Lasso' if regression else 'Logistic Regression'})")
    logger.info(f"    X_train: {X_train.shape}, y_train: {y_train.shape}")

    if regression:
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('model', Lasso(random_state=42))
        ])
        param_distributions = {
            'model__max_iter': [500, 1000, 2000, 5000],
            'model__alpha': np.logspace(-4, 2, 10)
        }
        param_grid = {
            'model__max_iter': [2000],
            'model__alpha': np.logspace(-3, 1, 5)
        }
        search_scoring = 'neg_mean_squared_error'
    else:
        n_classes = len(np.unique(y_train))
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(penalty='l1', solver='saga', random_state=42))
        ])
        param_distributions = {
            'model__max_iter': [500, 1000, 2000, 5000],
            'model__C': 1.0 / np.logspace(-4, 2, 10)
        }
        param_grid = {
            'model__max_iter': [2000],
            'model__C': 1.0 / np.logspace(-3, 1, 5)
        }
        search_scoring = 'roc_auc' if n_classes == 2 else 'f1_weighted'

    if search_type == "grid":
        search = GridSearchCV(
            model,
            param_grid,
            cv=3,
            scoring=search_scoring,
            n_jobs=-1,
            verbose=1
        )
        start_time = time.time()
        search.fit(X_train, y_train, model__sample_weight=sample_weights)
        logger.info(f"    Search completed in {time.time() - start_time:.2f}s")
        return search.best_params_, search.best_estimator_

    elif search_type == "random":
        search = RandomizedSearchCV(
            model,
            param_distributions,
            cv=5,
            scoring=search_scoring,
            n_jobs=-1,
            n_iter=5,
            random_state=42,
            verbose=1
        )
        start_time = time.time()
        search.fit(X_train, y_train, model__sample_weight=sample_weights)
        logger.info(f"    Search completed in {time.time() - start_time:.2f}s")
        return search.best_params_, search.best_estimator_

    elif search_type == "none":
        params = {"model__alpha": 1.0} if regression else {"model__C": 1.0}
        logger.info(f"    Using fixed parameters: {params}")
        model.set_params(**params)
        model.fit(X_train, y_train, model__sample_weight=sample_weights)
        return params, model

    else:
        raise ValueError(f"search_type must be 'grid', 'random', or 'none', got: {search_type}")


def train_tabpfn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    search_type: str = "none",
    sample_weights: Optional[np.ndarray] = None,
    task_type: str = "classification"
) -> Tuple[Dict, Any]:
    """
    Train TabPFN model (no hyperparameter search).

    Args:
        X_train: Training features
        y_train: Training labels
        search_type: Ignored (TabPFN doesn't need tuning)
        sample_weights: Ignored (TabPFN doesn't support sample weights)
        task_type: 'classification' or 'regression'

    Returns:
        Tuple of (empty_params_dict, trained_model)
    """
    if not TABPFN_AVAILABLE:
        raise ImportError("TabPFN not available. Install with: pip install tabpfn")

    regression = (task_type == "regression")
    logger.info(f"  Training TabPFN ({'Regression' if regression else 'Classification'})")
    logger.info(f"    X_train: {X_train.shape}, y_train: {y_train.shape}")

    if regression:
        model = TabPFNRegressor(ignore_pretraining_limits=True)
    else:
        model = TabPFNClassifier(ignore_pretraining_limits=True)

    start_time = time.time()
    model.fit(X_train, y_train)
    logger.info(f"    Training completed in {time.time() - start_time:.2f}s")

    return {}, model


# ==============================================================================
# MODEL REGISTRY
# Register new methods here - that's all you need to do!
# ==============================================================================

MODEL_REGISTRY = {
    'random_forest': train_random_forest,
    'xgboost': train_xgboost,
    'linear': train_linear,
    'tabpfn': train_tabpfn,
}


def get_available_methods() -> list:
    """Get list of all available methods."""
    return list(MODEL_REGISTRY.keys())


def train_model(
    method_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    search_type: str = "none",
    sample_weights: Optional[np.ndarray] = None,
    task_type: str = "classification"
) -> Tuple[Dict, Any]:
    """
    Train a model using the specified method.

    Args:
        method_name: Name of method (must be in MODEL_REGISTRY)
        X_train: Training features
        y_train: Training labels
        search_type: 'grid', 'random', or 'none'
        sample_weights: Optional sample weights
        task_type: 'classification' or 'regression'

    Returns:
        Tuple of (best_params, best_model)

    Raises:
        ValueError: If method_name not in MODEL_REGISTRY
    """
    if method_name not in MODEL_REGISTRY:
        available = ', '.join(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown method: '{method_name}'. "
            f"Available methods: {available}"
        )

    train_func = MODEL_REGISTRY[method_name]
    return train_func(X_train, y_train, search_type, sample_weights, task_type)


# ==============================================================================
# MODEL I/O UTILITIES
# ==============================================================================

def save_model(model_name: str, output_dir: Path, params: Dict, model: Any) -> None:
    """Save model and parameters to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save parameters as JSON
    params_path = output_dir / f"{model_name}_params.json"
    with open(params_path, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        params_serializable = {
            k: (v.item() if hasattr(v, 'item') else v)
            for k, v in params.items()
        }
        json.dump(params_serializable, f, indent=2)

    # Save model
    model_path = output_dir / f"{model_name}_model.pkl"
    joblib.dump(model, model_path)

    logger.info(f"    Saved model to {model_path}")


def load_model(model_name: str, output_dir: Path) -> Tuple[Dict, Any]:
    """Load model and parameters from disk."""
    params_path = output_dir / f"{model_name}_params.json"
    model_path = output_dir / f"{model_name}_model.pkl"

    with open(params_path, 'r') as f:
        params = json.load(f)

    model = joblib.load(model_path)

    return params, model


def model_exists(model_name: str, output_dir: Path) -> bool:
    """Check if model exists on disk."""
    params_path = output_dir / f"{model_name}_params.json"
    model_path = output_dir / f"{model_name}_model.pkl"

    return params_path.exists() and model_path.exists()


# ==============================================================================
# HOW TO ADD A NEW METHOD
# ==============================================================================
"""
To add a new ML method:

1. Write a training function following this signature:

   def train_yourmethod(
       X_train: np.ndarray,
       y_train: np.ndarray,
       search_type: str = "none",
       sample_weights: Optional[np.ndarray] = None,
       task_type: str = "classification"
   ) -> Tuple[Dict, Any]:
       # Your training logic here
       return params_dict, trained_model

2. Register it in MODEL_REGISTRY:

   MODEL_REGISTRY = {
       ...
       'yourmethod': train_yourmethod,
   }

3. That's it! Now you can use it in config:

   methods:
     yourmethod:
       search_type: "none"
       task_type: "classification"
"""

