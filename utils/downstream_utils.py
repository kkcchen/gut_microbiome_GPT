"""
Downstream Task Evaluation Orchestration
Handles configuration, data loading, and task execution.
All methods are now global (not per-task).
"""

import json
import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from omegaconf import OmegaConf, DictConfig
from scipy.stats import t as student_t
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.utils.class_weight import compute_class_weight

from trainers import logger
from .downstream_eval_utils import evaluate_multiclass_and_save, evaluate_regression_and_save
from .downstream_data_utils import handle_normalization_and_thresholding
from .downstream_summary_utils import write_downstream_summary
from .downstream_split_mode import (
    task_studies, resolve_split_mode, assert_disjoint, drop_rare_classes, resolve_cv_config,
)

# Import model training functions
from .downstream_models_utils import (
    train_model,
    save_model,
    load_model,
    model_exists,
    get_available_methods
)


# ==============================================================================
# CONFIGURATION & VALIDATION
# ==============================================================================

def validate_config(cfg: DictConfig) -> bool:
    """
    Validate configuration for downstream task evaluation.
    
    Args:
        cfg: OmegaConf configuration object
    
    Returns:
        True if config is valid, False otherwise
    """
    
    if 'downstream_tasks_config' not in cfg:
        logger.warning("No 'downstream_tasks_config' found in configuration")
        return False
    
    if not cfg.downstream_tasks_config.get('tasks'):
        logger.warning("No tasks defined in 'downstream_tasks_config.tasks'")
        return False
    
    if not cfg.downstream_tasks_config.get('methods'):
        logger.warning("No methods defined in 'downstream_tasks_config.methods'")
        return False

    return True


def get_cv_config(cfg: DictConfig) -> Dict:
    """cv defaults (outer_folds=5, min_samples_per_class=outer_folds) for pooled CV tasks."""
    cv_dict = cfg.downstream_tasks_config.get('cv', {})
    return resolve_cv_config(OmegaConf.to_container(cv_dict, resolve=True) if cv_dict else {})


def cv_for(cv_config: Dict, task_config: Dict) -> Dict:
    """cv settings in force for one task, after its per-task min_samples_per_class override."""
    return {**cv_config,
            "min_samples_per_class": task_config.get("min_samples_per_class",
                                                       cv_config["min_samples_per_class"])}


def get_embedding_paths(cfg: DictConfig) -> Tuple[Path, Path]:
    """
    Construct paths to saved embeddings based on eval_files in config.
    
    Args:
        cfg: OmegaConf configuration object
    
    Returns:
        Tuple of (train_embed_path, test_embed_path)
    """
    eval_files = cfg.paths.eval_files
    
    if len(eval_files) != 2:
        raise ValueError(
            f"Expected exactly 2 eval_files (train, test), got {len(eval_files)}. "
            f"Files: {eval_files}"
        )
    
    output_dir = Path(cfg.paths.output_dir)
    train_file_name = Path(eval_files[0]).stem
    test_file_name = Path(eval_files[1]).stem
    
    train_embed_path = output_dir / f"{train_file_name}.h5ad"
    test_embed_path = output_dir / f"{test_file_name}.h5ad"
    
    return train_embed_path, test_embed_path


def get_global_methods(cfg: DictConfig) -> Dict[str, Dict]:
    """
    Get global methods configuration and add task_type to each.
    
    Args:
        cfg: OmegaConf configuration object
    
    Returns:
        Dictionary of method configs
    """
    methods_config = OmegaConf.to_container(
        cfg.downstream_tasks_config.methods,
        resolve=True
    )
    return methods_config

# ==============================================================================
# EVALUATION
# ==============================================================================

def evaluate_and_save(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path,
    task_type: str,
    class_labels: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Evaluate predictions and save metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels/probabilities
        output_dir: Directory to save metrics
        task_type: 'classification' or 'regression'
        class_labels: Class labels for classification
    
    Returns:
        Dictionary of metrics
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if task_type == 'classification':
        return evaluate_multiclass_and_save(y_true, y_pred, class_labels, str(output_dir))
    else:
        return evaluate_regression_and_save(y_true, y_pred, str(output_dir))


# ==============================================================================
# TASK EXECUTION
# ==============================================================================

# Maps a search's sklearn `scoring=` name to the downstream metric key it corresponds to,
# so the inner-CV std reported by a presplit task's hyperparameter search (see
# run_single_method below) gets attached to the metric it actually optimized -- e.g.
# xgboost's multiclass search scores 'roc_auc_ovr_weighted', not F1, so it must map to
# "auroc" rather than "f1_weighted".
SCORING_TO_METRIC = {
    "roc_auc": "auroc",
    "roc_auc_ovr_weighted": "auroc",
    "f1_weighted": "f1_weighted",
}


def run_single_method(
    method_name: str,
    method_config: Dict,
    task_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: Path,
    sample_weights: Optional[np.ndarray] = None,
    random_state: int = 42
) -> bool:
    """
    Run a single method on a task.

    This function:
    1. Checks if model is already trained (cache)
    2. If not, calls train_model() from downstream_models_utils.py
    3. Saves the trained model
    4. Makes predictions and evaluates

    Args:
        method_name: Name of method
        method_config: Method configuration from YAML
        task_type: 'classification' or 'regression' (inferred from label_type)
        X_train, y_train: Training data
        X_test, y_test: Test data
        output_dir: Directory to save results
        sample_weights: Optional sample weights
        random_state: Seed for the model and any hyperparameter search

    Returns:
        True if successful, False otherwise
    """
    try:
        search_type = method_config.get('search_type', 'none')
        
        logger.info(f"\n  Method: {method_name}")
        logger.info(f"    Task type: {task_type}")
        logger.info(f"    Search type: {search_type}")
        
        # Create method output directory
        method_dir = output_dir / method_name / task_type
        method_dir.mkdir(parents=True, exist_ok=True)
        
        model_name = f"{task_type}_model"
        
        # Train or load model
        if model_exists(model_name, method_dir):
            logger.info(f"    Loading existing model...")
            params, model = load_model(model_name, method_dir)
            inner_std = params.pop("_inner_cv_std", None)
            search_scoring = params.pop("_search_scoring", None)
        else:
            logger.info(f"    Training new model...")

            # Call the training function from downstream_models_utils.py
            params, model, inner_std, search_scoring = train_model(
                method_name=method_name,
                X_train=X_train,
                y_train=y_train,
                search_type=search_type,
                sample_weights=sample_weights,
                task_type=task_type,
                random_state=random_state
            )

            # Save model. inner_std/search_scoring ride along inside the saved params
            # JSON (rather than a separate file) so a cached rerun (the branch above)
            # still has them to re-derive search_cv_std.json below.
            params_to_save = {**params, "_inner_cv_std": inner_std, "_search_scoring": search_scoring}
            save_model(model_name, method_dir, params_to_save, model)
        
        # Make predictions
        if task_type == 'regression':
            y_pred = model.predict(X_test)
        else:
            # For classification, get the original classes
            if hasattr(model, '_label_encoder'):
                # XGBoost with label encoder - need to handle encoded labels
                label_encoder = model._label_encoder
                class_labels = label_encoder.classes_
                
                # Handle class filtering for classification
                valid_classes = set(class_labels)
                mask_valid = y_test.isin(valid_classes) if hasattr(y_test, 'isin') else np.isin(y_test, list(valid_classes))
                
                if not mask_valid.all():
                    removed = set(y_test[~mask_valid])
                    logger.warning(f"    Removing {(~mask_valid).sum()} test samples with unseen labels: {removed}")
                    X_test = X_test[mask_valid]
                    y_test = y_test[mask_valid]
                
                # Get probability predictions (already in correct order from encoded classes)
                y_pred = model.predict_proba(X_test)
                
            elif hasattr(model, 'classes_'):
                # Other classifiers (RF, etc.) that have classes_ attribute
                class_labels = model.classes_
                
                # Handle class filtering
                valid_classes = set(class_labels)
                mask_valid = y_test.isin(valid_classes) if hasattr(y_test, 'isin') else np.isin(y_test, list(valid_classes))
                
                if not mask_valid.all():
                    removed = set(y_test[~mask_valid])
                    logger.warning(f"    Removing {(~mask_valid).sum()} test samples with unseen labels: {removed}")
                    X_test = X_test[mask_valid]
                    y_test = y_test[mask_valid]
                
                y_pred = model.predict_proba(X_test)
                
            else:
                # Fallback
                class_labels = np.unique(y_train)
                y_pred = model.predict_proba(X_test)
        
        # Evaluate
        logger.info(f"    Evaluating on {len(y_test)} test samples...")
        metrics = evaluate_and_save(
            y_test, y_pred, method_dir,
            task_type,
            class_labels if task_type == 'classification' else None
        )

        # Presplit tasks have no across-fold spread to quote (see run_task_combined_cv's
        # docstring for the pooled-CV counterpart that does), so the best this can report is
        # the inner search's own CV std at the winning hyperparameters -- a proxy, not a real
        # held-out error bar, matching kd_tasks/downstream_final_utils.py::run_presplit.
        # Classification only: there's no established way to translate a regression search's
        # neg_mean_squared_error std onto the reported MAE/RMSE/R2 scale.
        if task_type == "classification" and inner_std is not None:
            metric_key = SCORING_TO_METRIC.get(search_scoring)
            if metric_key is not None:
                with open(method_dir / "search_cv_std.json", "w") as f:
                    json.dump({"metric": metric_key, "std": inner_std,
                              "search_scoring": search_scoring}, f, indent=2)
            else:
                logger.warning(f"    No metric mapping for search_scoring={search_scoring!r}; "
                               f"skipping search_cv_std.json")

        logger.info(f"    ✓ Method {method_name} completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"    ✗ Method {method_name} failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False


# ==============================================================================
# POOLED CROSS-VALIDATION (single-study tasks)
#
# A task drawn from only one study has no natural train/test boundary -- see
# utils/downstream_split_mode.py::resolve_split_mode. For those tasks, run_single_task_cv
# pools train+test and runs outer StratifiedKFold/KFold; each method's own train_model()
# call already does a full hyperparameter search inside that outer fold's training rows
# only (search_type='grid'/'random'), so this outer loop is what makes it nested CV.
# Tasks with disjoint train/test studies (a real cross-study holdout) keep using
# run_single_method above, unchanged.
# ==============================================================================

def balanced_weights(y) -> np.ndarray:
    """Per-sample weights balancing y's classes, for sklearn's sample_weight= kwarg."""
    classes = np.unique(y)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    weights_dict = dict(zip(classes, class_weights))
    return np.array([weights_dict[label] for label in y])


def predict_proba_ordered(model, X: np.ndarray, classes: List) -> np.ndarray:
    """
    predict_proba with columns forced into `classes` order.

    XGBoost labels its columns by the LabelEncoder it fits internally; other classifiers
    (RandomForest, the linear pipeline, TabPFN) follow their own `classes_`. Reindexing
    keeps per-class metrics attached to the right class across every method and fold.
    """
    probs = model.predict_proba(X)
    if hasattr(model, "_label_encoder"):
        model_classes = list(model._label_encoder.classes_)
    else:
        model_classes = list(model.classes_)
    if model_classes == list(classes):
        return probs
    position = {c: i for i, c in enumerate(model_classes)}
    return probs[:, [position[c] for c in classes]]


def _classification_fold_metrics(y_true: np.ndarray, probs: np.ndarray, classes: List) -> Dict[str, float]:
    """accuracy/f1_macro/f1_weighted/auroc for one fold -- the same metric names
    utils/downstream_summary_utils.py already reports for the presplit path."""
    class_arr = np.asarray(classes, dtype=object)
    y_true = np.asarray(y_true, dtype=object)
    y_pred = class_arr[np.argmax(probs, axis=1)]

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    if len(classes) == 2:
        y_true_bin = (y_true == class_arr[1]).astype(int)
        metrics["auroc"] = roc_auc_score(y_true_bin, probs[:, 1])
    else:
        class_index = {c: i for i, c in enumerate(classes)}
        y_true_idx = np.array([class_index[v] for v in y_true])
        metrics["auroc"] = roc_auc_score(y_true_idx, probs, multi_class="ovr",
                                         average="weighted", labels=list(range(len(classes))))
    return metrics


def _regression_fold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
        "r2": r2_score(y_true, y_pred),
    }


def aggregate_fold_metrics(fold_metrics: List[Dict], confidence: float = 0.95) -> Dict:
    """Mean, std(ddof=1) and a t-based CI margin per metric, across outer folds."""
    out = {}
    n = len(fold_metrics)
    for key in fold_metrics[0]:
        values = np.array([m[key] for m in fold_metrics], dtype=float)
        mean, std = float(values.mean()), float(values.std(ddof=1))
        margin = float(student_t.ppf(0.5 + confidence / 2, n - 1) * std / np.sqrt(n))
        out[key] = {"mean": mean, "std": std, "ci_margin": margin, "n_folds": n,
                    "values": values.tolist()}
    return out


def run_single_method_cv(
    method_name: str,
    method_config: Dict,
    task_type: str,
    X: np.ndarray,
    y: pd.Series,
    splitter,
    classes: Optional[List],
    output_dir: Path,
    random_state: int = 42,
) -> bool:
    """
    Pooled-CV counterpart to run_single_method: an outer fold loop around the same
    train_model()/evaluate_and_save() calls, writing a fold_<i>/ subdirectory per fold (so
    the existing fixed evaluate_and_save filenames never collide) plus one aggregate
    cv_summary.json per method with the fold mean/std/CI for each metric.
    """
    try:
        method_dir = output_dir / method_name / task_type
        method_dir.mkdir(parents=True, exist_ok=True)
        summary_path = method_dir / "cv_summary.json"

        if summary_path.exists():
            logger.info(f"\n  Method: {method_name} ({task_type}, pooled CV) - cached")
            return True

        n_splits = splitter.get_n_splits()
        logger.info(f"\n  Method: {method_name}")
        logger.info(f"    Task type: {task_type} (pooled {n_splits}-fold CV), X: {X.shape}")

        y_values = y.to_numpy()
        fold_metrics, fold_params = [], []

        for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y_values)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            sample_weights = balanced_weights(y_train) if task_type == "classification" else None

            # inner_std/search_scoring unused here -- this path already has a real
            # per-fold spread via aggregate_fold_metrics below.
            params, model, _inner_std, _search_scoring = train_model(
                method_name=method_name,
                X_train=X_train,
                y_train=y_train,
                search_type=method_config.get("search_type", "none"),
                sample_weights=sample_weights,
                task_type=task_type,
                random_state=random_state,
            )

            fold_dir = method_dir / f"fold_{fold_idx}"
            if task_type == "regression":
                y_pred = model.predict(X_test)
                metrics = _regression_fold_metrics(y_test.to_numpy(), y_pred)
                evaluate_and_save(y_test, y_pred, fold_dir, task_type, None)
            else:
                probs = predict_proba_ordered(model, X_test, classes)
                metrics = _classification_fold_metrics(y_test.to_numpy(), probs, classes)
                evaluate_and_save(y_test, probs, fold_dir, task_type, np.array(classes, dtype=object))

            fold_metrics.append(metrics)
            fold_params.append(params)
            logger.info(f"    fold {fold_idx + 1}/{n_splits} (n_test={len(test_idx)}): "
                       f"{ {k: round(v, 4) for k, v in metrics.items()} }")

        aggregates = aggregate_fold_metrics(fold_metrics)
        payload = {
            "eval_mode": "combined_cv",
            "outer_folds": n_splits,
            "aggregates": aggregates,
            "fold_params": fold_params,
            "n_samples": len(y),
        }
        if task_type == "classification":
            payload["classes"] = [str(c) for c in classes]

        with open(summary_path, "w") as f:
            json.dump(payload, f, indent=2, default=str)

        logger.info(f"    {method_name}: " +
                   ", ".join(f"{k}={v['mean']:.4f}+/-{v['ci_margin']:.4f}"
                             for k, v in aggregates.items()))
        logger.info(f"    ✓ Method {method_name} completed successfully")
        return True

    except Exception as e:
        logger.error(f"    ✗ Method {method_name} (pooled CV) failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def run_task_combined_cv(
    task_name: str,
    task_type: str,
    label_type: str,
    train_task: ad.AnnData,
    test_task: ad.AnnData,
    global_methods: Dict[str, Dict],
    output_dir: Path,
    cv_config: Dict,
    random_state: int = 42,
) -> int:
    """
    Pool a single-study task's train+test rows and run every method's outer-fold CV.

    :return: number of methods that completed successfully.
    """
    assert_disjoint(train_task.obs_names, test_task.obs_names, task_name)

    labels_all = pd.concat([train_task.obs[label_type], test_task.obs[label_type]]).astype(object)
    X_all = np.concatenate([
        np.asarray(train_task.X, dtype=np.float32),
        np.asarray(test_task.X, dtype=np.float32),
    ])
    keep = labels_all.notna().to_numpy()
    y = labels_all[keep].reset_index(drop=True)
    X = X_all[keep]

    if task_type == "classification":
        y, dropped = drop_rare_classes(y, cv_config["min_samples_per_class"], task_name)
        X = X[y.index.to_numpy()]
        if y.nunique() < 2:
            logger.warning(f"  '{task_name}': fewer than 2 usable classes after pooling, skipping.")
            return 0
        classes = sorted(y.unique().tolist(), key=str)
        splitter = StratifiedKFold(n_splits=cv_config["outer_folds"], shuffle=True,
                                   random_state=random_state)
        logger.info(f"  Pooled: {X.shape}, classes={classes}"
                   + (f", dropped={dropped}" if dropped else ""))
    else:
        classes = None
        splitter = KFold(n_splits=cv_config["outer_folds"], shuffle=True, random_state=random_state)
        logger.info(f"  Pooled: {X.shape}")

    success_count = 0
    for method_name, method_config in global_methods.items():
        if run_single_method_cv(method_name, method_config, task_type, X, y, splitter,
                                classes, output_dir, random_state):
            success_count += 1
    return success_count


def process_single_task(
    task_name: str,
    task_config: Dict[str, Any],
    global_methods: Dict[str, Dict],
    adata_train: ad.AnnData,
    adata_test: ad.AnnData,
    output_dir: Path,
    cv_config: Optional[Dict] = None,
    random_state: int = 42
) -> bool:
    """
    Process a single downstream task with all global methods.

    Whether this task is presplit (train on train, evaluate on test) or pooled+cross-validated
    depends on how many distinct obs['study_id'] values it spans -- see
    utils/downstream_split_mode.py::resolve_split_mode. A task can override the auto-detected
    mode with an explicit task_config.split_mode of "combined_cv" or "presplit".

    Args:
        task_name: Name of the task
        task_config: Task configuration from YAML
        global_methods: Global methods config (same for all tasks)
        adata_train: Training AnnData with embeddings
        adata_test: Test AnnData with embeddings
        output_dir: Base output directory
        cv_config: outer_folds/min_samples_per_class for pooled CV tasks (see get_cv_config);
            defaults applied via resolve_cv_config if not given
        random_state: Seed for the model, any hyperparameter search, and (for pooled tasks)
            the outer fold split

    Returns:
        True if at least one method succeeded, False otherwise
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Task: {task_name}")
    logger.info(f"{'='*80}")

    try:
        # Determine task type
        label_type = task_config.label_type
        task_type = 'regression' if label_type == 'continuous_label' else 'classification'

        logger.info(f"Label type: {label_type}")
        logger.info(f"Task type: {task_type}")
        logger.info(f"Target column: {task_config.target_col}")
        logger.info(f"Ignored labels: {task_config.ignored_labels}")
        logger.info(f"Methods: {list(global_methods.keys())}")
        target_col = task_config.target_col
        ignored_labels = task_config.ignored_labels

        # Create task output directory
        task_dir = output_dir / task_name
        task_dir.mkdir(parents=True, exist_ok=True)

        studies = task_studies({"train": adata_train.obs, "test": adata_test.obs},
                               target_col, task_name, label_type, ignored_labels)
        mode = resolve_split_mode(studies, task_config.get("split_mode", "auto"))
        logger.info(f"Split mode: {mode} ({len(studies)} distinct study/studies: {studies})")

        # Filter data for this task
        train_task = adata_train[adata_train.obs[target_col] == task_name].copy()
        test_task = adata_test[adata_test.obs[target_col] == task_name].copy()

        # Remove ignored labels
        if len(ignored_labels) > 0:
            train_task = train_task[~train_task.obs[label_type].isin(ignored_labels)]
            test_task = test_task[~test_task.obs[label_type].isin(ignored_labels)]

        embed_dir = task_dir

        if mode == "combined_cv":
            cv = cv_for(cv_config or resolve_cv_config(None), task_config)
            success_count = run_task_combined_cv(
                task_name, task_type, label_type, train_task, test_task,
                global_methods, embed_dir, cv, random_state)
        else:
            # Extract features and labels
            X_train = np.asarray(train_task.X, dtype=np.float32)
            y_train = train_task.obs[label_type]
            X_test = np.asarray(test_task.X, dtype=np.float32)
            y_test = test_task.obs[label_type]

            logger.info(f"  Train: {X_train.shape}, Test: {X_test.shape}")

            # Compute sample weights for classification
            sample_weights = None
            if task_type == 'classification':
                sample_weights = balanced_weights(y_train)
                logger.info(f"  Using balanced class weights")

            # Run each global method
            success_count = 0
            for method_name, method_config in global_methods.items():
                method_success = run_single_method(
                    method_name,
                    method_config,
                    task_type,
                    X_train, y_train,
                    X_test, y_test,
                    embed_dir,
                    sample_weights,
                    random_state
                )

                if method_success:
                    success_count += 1

        logger.info(f"\n Task {task_name} completed ({success_count} method(s) succeeded)")
        return success_count > 0

    except Exception as e:
        logger.error(f"\n Task {task_name} failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def run_downstream_evaluation(
    cfg: DictConfig,
    accelerator: Any,
    skip_if_exists: bool = False
) -> None:
    """
    Main function to run downstream task evaluation.
    
    This is the ONLY function that should be called from inference.py
    
    Workflow:
    1. Validate configuration
    2. Load embeddings from h5ad files
    3. Get global methods configuration
    4. For each task, call process_single_task()
    5. Print summary
    
    Args:
        cfg: OmegaConf configuration object (YAML-based)
        accelerator: Accelerate accelerator instance
        skip_if_exists: If True, skip tasks that already have results
    """
    # Validate configuration
    if not validate_config(cfg):
        return
    
    logger.info("=" * 80)
    logger.info("STARTING DOWNSTREAM TASK EVALUATION")
    logger.info("=" * 80)
    logger.info(f"Available methods: {', '.join(get_available_methods())}")
    
    # Get embedding paths
    try:
        if cfg.eval.with_model:
            train_embed_path, test_embed_path = get_embedding_paths(cfg)
        else:
            logger.info("Skipping embedding loading (with_model=False), using raw data for downstream tasks")
            train_embed_path, test_embed_path = None, None
            train_embed_path = Path(cfg.paths.train_path)
            test_embed_path = Path(cfg.paths.test_path)
            if not train_embed_path.exists() or not test_embed_path.exists():
                logger.error(f"Raw data files not found at {train_embed_path} and {test_embed_path}")
                return
    except ValueError as e:
        logger.error(str(e))
        return
    
    # Wait for all processes to finish saving embeddings
    accelerator.wait_for_everyone()
    
    # Only run on main process
    if not accelerator.is_main_process:
        logger.info("Skipping downstream evaluation on non-main process")
        return
    
    # Verify embedding files exist
    if not train_embed_path.exists():
        logger.error(f"Training embeddings not found at {train_embed_path}")
        return
    if not test_embed_path.exists():
        logger.error(f"Test embeddings not found at {test_embed_path}")
        return
    
    # Load embeddings
    logger.info(f"Loading train embeddings from {train_embed_path}")
    adata_train = ad.read_h5ad(train_embed_path)
    logger.info(f"  Train: {adata_train.shape[0]} samples")
    
    logger.info(f"Loading test embeddings from {test_embed_path}")
    adata_test = ad.read_h5ad(test_embed_path)
    logger.info(f"  Test: {adata_test.shape[0]} samples")

    # handle normalization and thresholding
    adata_train, adata_test = handle_normalization_and_thresholding(adata_train, adata_test, cfg)
    
    # Create output directory
    output_dir = Path(cfg.paths.output_dir) / "downstream_tasks"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to: {output_dir}")
    
    # Get global methods configuration
    global_methods = get_global_methods(cfg)
    logger.info(f"\nGlobal methods: {list(global_methods.keys())}")

    # cv settings for single-study tasks (pooled + cross-validated)
    cv_config = get_cv_config(cfg)
    logger.info(f"CV config (single-study tasks): {cv_config}")

    # Convert tasks config to dict
    tasks_config = cfg.downstream_tasks_config.tasks
    # Process each task
    total_tasks = len(tasks_config)
    successful_tasks = 0
    failed_tasks = []
    
    logger.info(f"\nProcessing {total_tasks} downstream tasks...")
    
    for i, task_name in enumerate(tasks_config.keys(), 1):
        task_config = tasks_config[task_name]
        logger.info(f"\n[{i}/{total_tasks}] Processing: {task_name}")
        
        # Check if task already exists
        task_output_path = output_dir / task_name
        if skip_if_exists and task_output_path.exists():
            logger.info(f"  Skipping {task_name} (results already exist)")
            successful_tasks += 1
            continue
        
        # Process task with global methods
        success = process_single_task(
            task_name=task_name,
            task_config=task_config,
            global_methods=global_methods,
            adata_train=adata_train,
            adata_test=adata_test,
            output_dir=output_dir,
            cv_config=cv_config,
            random_state=cfg.eval.get('seed', 42)
        )
        
        if success:
            successful_tasks += 1
        else:
            failed_tasks.append(task_name)

        # re-written after every task (not just at the end) so a hard kill partway
        # through a long eval job still leaves an accurate, current summary
        write_downstream_summary(Path(cfg.paths.output_dir))

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("DOWNSTREAM TASK EVALUATION COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Total tasks: {total_tasks}")
    logger.info(f"Successful: {successful_tasks}")
    logger.info(f"Failed: {len(failed_tasks)}")
    
    if failed_tasks:
        logger.info(f"\nFailed tasks: {', '.join(failed_tasks)}")
    
    logger.info(f"\nResults saved to: {output_dir}")
    logger.info("=" * 80)
