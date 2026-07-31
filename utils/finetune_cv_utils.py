"""
Pooled K-fold cross-validation for single-study finetuning tasks.

Mirrors utils/downstream_utils.py::run_task_combined_cv for the classical-ML pipeline, but
for neural-net finetuning: there's no cheap in-memory reuse option here, every fold is a
full training run via scripts/finetune.py, so this module materializes small per-fold
train/test h5ad files (the pattern scripts/preprocess.py already uses for writing subset
AnnData objects) and lets utils/finetune_orchestration.py drive scripts.finetune.main(cfg)
against them unchanged, one config per fold.

No inner hyperparameter search exists in finetuning (hyperparams are fixed by config), so
this is plain K-fold CV, not nested -- still the right fix for the same reason
utils/downstream_split_mode.py exists: one fixed train/test split on a single-study task is
a fragile point estimate, not a real accuracy measurement.

Critical ordering invariant: rare classes are dropped from the *pooled* label distribution
before folds are built, with a floor of min_samples_per_class >= outer_folds (see
downstream_split_mode.drop_rare_classes). That guarantees every surviving class appears in
every fold's train AND test rows, which two things downstream depend on without any further
change: utils/data_pipeline.py's LabelEncoder (fit on a fold's train file, applied to its test
file) never sees an unseen test label, and trainers/trainer.py::evaluate_on_test_set's
binary-vs-multiclass branch (driven by which classes are present in the current test batch)
produces the same metric key set in every fold, which aggregate_fold_metrics needs to line up.
"""
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
from omegaconf import OmegaConf
from scipy.stats import t as student_t
from sklearn.model_selection import KFold, StratifiedKFold

from .downstream_split_mode import (
    assert_disjoint, drop_rare_classes, read_obs, resolve_split_mode, task_studies,
)

TARGET_COLNAME = "downstream_task"


def resolve_task_mode(base_train_h5ad: str, base_test_h5ad: str, task_name: str,
                      label_col: str, ignored_labels, declared: str = "auto") -> Tuple[str, List[str]]:
    """Cheap obs-only read of the fixed base h5ads to decide one task's split mode."""
    obs_frames = {"train": read_obs(base_train_h5ad), "test": read_obs(base_test_h5ad)}
    studies = task_studies(obs_frames, TARGET_COLNAME, task_name, label_col, ignored_labels)
    return resolve_split_mode(studies, declared), studies


def materialize_pooled_task_adata(adata_train: ad.AnnData, adata_test: ad.AnnData,
                                  task_name: str, label_col: str, ignored_labels) -> ad.AnnData:
    """Filter both splits to this task, drop unlabeled/ignored rows, and pool them."""
    def _task_rows(adata):
        mask = ((adata.obs[TARGET_COLNAME] == task_name)
                & adata.obs[label_col].notna()
                & ~adata.obs[label_col].isin(ignored_labels))
        return adata[mask].copy()

    train_task = _task_rows(adata_train)
    test_task = _task_rows(adata_test)
    assert_disjoint(train_task.obs_names, test_task.obs_names, task_name)
    return ad.concat([train_task, test_task])


def build_folds(pooled: ad.AnnData, label_col: str, finetune_task: str, task_name: str,
                outer_folds: int, min_samples_per_class: int, seed: int
                ) -> Tuple[List[Tuple[ad.AnnData, ad.AnnData]], Dict]:
    """
    Outer StratifiedKFold (classification, after dropping classes too rare to stratify) or
    plain KFold (regression).

    :return: (fold_pairs, dropped_classes) where fold_pairs[i] = (fold_train_adata, fold_test_adata).
    """
    dropped = {}
    if finetune_task == "classification":
        labels = pooled.obs[label_col].astype(object)
        labels, dropped = drop_rare_classes(labels, min_samples_per_class, task_name)
        pooled = pooled[labels.index].copy()
        splitter = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=seed)
        split_iter = splitter.split(np.zeros(len(labels)), labels.to_numpy())
    else:
        splitter = KFold(n_splits=outer_folds, shuffle=True, random_state=seed)
        split_iter = splitter.split(np.zeros(pooled.n_obs))

    return [(pooled[train_idx].copy(), pooled[test_idx].copy())
            for train_idx, test_idx in split_iter], dropped


def write_fold_h5ads(fold_pairs: List[Tuple[ad.AnnData, ad.AnnData]], out_dir: str) -> List[Tuple[str, str]]:
    """Write out_dir/fold_<i>/{downstream_train,downstream_test}.h5ad, one pair per fold."""
    paths = []
    for i, (fold_train, fold_test) in enumerate(fold_pairs):
        fold_dir = Path(out_dir) / f"fold_{i}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_path, test_path = fold_dir / "downstream_train.h5ad", fold_dir / "downstream_test.h5ad"
        fold_train.write_h5ad(train_path)
        fold_test.write_h5ad(test_path)
        paths.append((str(train_path), str(test_path)))
    return paths


def aggregate_fold_metrics(fold_metrics: List[Dict[str, float]], confidence: float = 0.95) -> Dict:
    """Mean, std(ddof=1) and a t-based CI margin per metric key, across folds."""
    out = {}
    keys = set().union(*(m.keys() for m in fold_metrics)) if fold_metrics else set()
    for key in keys:
        values = np.array([m[key] for m in fold_metrics if key in m], dtype=float)
        n = len(values)
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if n > 1 else 0.0
        margin = float(student_t.ppf(0.5 + confidence / 2, n - 1) * std / np.sqrt(n)) if n > 1 else 0.0
        out[key] = {"mean": mean, "std": std, "ci_margin": margin, "n_folds": n, "values": values.tolist()}
    return out


def _read_fold_test_metrics(fold_output_dir: str) -> Optional[Dict[str, float]]:
    """Every numeric key from one fold's best_model/test_metrics.yaml (as written by
    trainers/trainer.py::evaluate_on_test_set), unfiltered -- which keys are present
    (test_f1 vs test_f1_macro etc.) depends on binary-vs-multiclass, and is guaranteed
    identical across folds by the class-floor invariant documented at module level."""
    metrics_path = Path(fold_output_dir) / "best_model" / "test_metrics.yaml"
    if not metrics_path.exists():
        return None
    metrics = OmegaConf.to_container(OmegaConf.load(metrics_path), resolve=True)
    return {k: v for k, v in metrics.items() if isinstance(v, (int, float))}


def run_task_cv(
    pretrain_cfg,
    pretrain_output_dir: str,
    task_name: str,
    task_spec: Dict,
    finetune_output_root: Optional[str],
    cv_conf: Dict,
    seed: int,
    adata_train_full: ad.AnnData,
    adata_test_full: ad.AnnData,
    build_finetune_config,
    run_single_finetune,
) -> Dict:
    """
    Pool one single-study task's train+test rows, run outer-fold CV, aggregate.

    build_finetune_config/run_single_finetune are passed in rather than imported, to avoid a
    circular import with utils/finetune_orchestration.py (where they live and where this is
    called from).

    :return: {"status": "ok", "output_dir": <task-level dir holding cv_summary.yaml>}
    """
    result_root = finetune_output_root if finetune_output_root is not None else pretrain_output_dir
    task_output_dir = os.path.join(result_root, "finetune", task_name)
    os.makedirs(task_output_dir, exist_ok=True)

    label_col = task_spec["label_column"]
    ignored_labels = task_spec.get("ignored_labels", [])
    finetune_task = task_spec["finetune_task"]

    pooled = materialize_pooled_task_adata(adata_train_full, adata_test_full, task_name,
                                           label_col, ignored_labels)
    fold_pairs, dropped = build_folds(pooled, label_col, finetune_task, task_name,
                                      cv_conf["outer_folds"], cv_conf["min_samples_per_class"], seed)

    cv_data_dir = os.path.join(task_output_dir, "cv_data")
    fold_h5ad_paths = write_fold_h5ads(fold_pairs, cv_data_dir)

    fold_metrics, fold_output_dirs = [], []
    for fold_idx, fold_paths in enumerate(fold_h5ad_paths):
        fold_cfg = build_finetune_config(
            pretrain_cfg, pretrain_output_dir, task_name, task_spec,
            finetune_output_root=finetune_output_root, fold_paths=fold_paths, fold_idx=fold_idx,
        )
        run_single_finetune(fold_cfg)
        metrics = _read_fold_test_metrics(fold_cfg.paths.output_dir)
        if metrics is None:
            raise RuntimeError(f"fold {fold_idx} of task '{task_name}' did not produce "
                               f"test_metrics.yaml at {fold_cfg.paths.output_dir}")
        fold_metrics.append(metrics)
        fold_output_dirs.append(fold_cfg.paths.output_dir)

    if not cv_conf.get("keep_fold_data", False):
        shutil.rmtree(cv_data_dir, ignore_errors=True)

    payload = {
        "eval_mode": "combined_cv",
        "outer_folds": cv_conf["outer_folds"],
        "n_samples": pooled.n_obs,
        "dropped_classes": dropped,
        "aggregates": aggregate_fold_metrics(fold_metrics),
        "fold_output_dirs": fold_output_dirs,
    }
    OmegaConf.save(OmegaConf.create(payload), Path(task_output_dir) / "cv_summary.yaml")

    return {"status": "ok", "output_dir": task_output_dir}
