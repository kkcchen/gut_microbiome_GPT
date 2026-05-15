"""
Downstream Task Evaluation Orchestration
Handles configuration, data loading, and task execution.
All methods are now global (not per-task).
"""

import numpy as np
import anndata as ad
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from omegaconf import OmegaConf, DictConfig
from sklearn.utils.class_weight import compute_class_weight

from trainers import logger
from .downstream_eval_utils import evaluate_multiclass_and_save, evaluate_regression_and_save
from .downstream_data_utils import handle_normalization_and_thresholding

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

def run_single_method(
    method_name: str,
    method_config: Dict,
    task_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: Path,
    sample_weights: Optional[np.ndarray] = None
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
        else:
            logger.info(f"    Training new model...")
            
            # Call the training function from downstream_models_utils.py
            params, model = train_model(
                method_name=method_name,
                X_train=X_train,
                y_train=y_train,
                search_type=search_type,
                sample_weights=sample_weights,
                task_type=task_type
            )
            
            # Save model
            save_model(model_name, method_dir, params, model)
        
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
        
        logger.info(f"    ✓ Method {method_name} completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"    ✗ Method {method_name} failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def process_single_task(
    task_name: str,
    task_config: Dict[str, Any],
    global_methods: Dict[str, Dict],
    adata_train: ad.AnnData,
    adata_test: ad.AnnData,
    output_dir: Path
) -> bool:
    """
    Process a single downstream task with all global methods.
    
    Args:
        task_name: Name of the task
        task_config: Task configuration from YAML
        global_methods: Global methods config (same for all tasks)
        adata_train: Training AnnData with embeddings
        adata_test: Test AnnData with embeddings
        output_dir: Base output directory
    
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
        
        success_count = 0
        
            
        # Filter data for this task
        train_task = adata_train[adata_train.obs[target_col] == task_name].copy()
        test_task = adata_test[adata_test.obs[target_col] == task_name].copy()
        
        # Remove ignored labels
        if len(ignored_labels) > 0:
            train_task = train_task[~train_task.obs[label_type].isin(ignored_labels)]
            test_task = test_task[~test_task.obs[label_type].isin(ignored_labels)]
        
        # Extract features and labels
        X_train = np.asarray(train_task.X, dtype=np.float32)
        y_train = train_task.obs[label_type]
        X_test = np.asarray(test_task.X, dtype=np.float32)
        y_test = test_task.obs[label_type]
          
        
        logger.info(f"  Train: {X_train.shape}, Test: {X_test.shape}")
        
        # Compute sample weights for classification
        sample_weights = None
        if task_type == 'classification':
            classes = np.unique(y_train)
            class_weights = compute_class_weight(
                class_weight="balanced",
                classes=classes,
                y=y_train
            )
            class_weights_dict = dict(zip(classes, class_weights))
            sample_weights = np.array([class_weights_dict[label] for label in y_train])
            logger.info(f"  Using balanced class weights")
        
        # Create embedding-specific output directory
        embed_dir = task_dir
        
        # Run each global method
        for method_name, method_config in global_methods.items():
            method_success = run_single_method(
                method_name,
                method_config,
                task_type,
                X_train, y_train,
                X_test, y_test,
                embed_dir,
                sample_weights
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
            output_dir=output_dir
        )
        
        if success:
            successful_tasks += 1
        else:
            failed_tasks.append(task_name)
    
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
