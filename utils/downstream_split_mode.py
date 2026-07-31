"""
Shared single-study-vs-multi-study detection for downstream task evaluation.

Ported out of kd_tasks/downstream_final_utils.py so both the classical-ML embedding-
baselines pipeline (utils/downstream_utils.py) and the neural-net finetuning pipeline
(utils/finetune_orchestration.py, utils/finetune_cv_utils.py) share one definition of
"does this task span more than one study" instead of drifting apart.

Whether a task should be pooled (train+test combined, then cross-validated) or kept as a
presplit train/test pair depends on obs['study_id']: a task drawn from a single study has
no natural train/test boundary, so one fixed holdout is a fragile point estimate -- pooling
and cross-validating gives a mean with an error margin instead. A task whose train and test
studies are disjoint (currently `location` and `sex`) is a genuine cross-study holdout, and
reshuffling it into folds would destroy exactly what it measures, so those stay presplit.
This is decided by counting distinct studies in the data, not by a hardcoded task list, so
it stays correct if the underlying data changes.
"""

import logging
from typing import Dict, List

import h5py
import pandas as pd
from anndata import io as ad_io

STUDY_COLNAME = "study_id"

logger = logging.getLogger("downstream_split_mode")


def read_obs(h5ad_path: str) -> pd.DataFrame:
    """Read .obs without touching X -- cheap enough to call just to decide a task's mode."""
    with h5py.File(h5ad_path, "r") as f:
        return ad_io.read_elem(f["obs"])


def task_studies(obs_frames: Dict[str, pd.DataFrame], target_col: str, task_name: str,
                 label_col: str, ignored_labels) -> List[str]:
    """
    Distinct study_id values across every split's rows for one task, after dropping rows
    with no label or an ignored label -- the same rows that would actually be trained/
    evaluated on, so the study count reflects usable data, not raw presence in the file.
    """
    studies = []
    for obs in obs_frames.values():
        mask = ((obs[target_col] == task_name)
                & obs[label_col].notna()
                & ~obs[label_col].isin(ignored_labels))
        if STUDY_COLNAME in obs:
            studies += obs.loc[mask, STUDY_COLNAME].astype(object).unique().tolist()
    return sorted(set(studies))


def resolve_split_mode(studies: List[str], declared: str = "auto") -> str:
    """
    "combined_cv" -- pool the splits and cross-validate (single-study tasks)
    "presplit"    -- train on train, evaluate on test (disjoint-study tasks)

    Auto-detection counts distinct study_id values; declared overrides it explicitly.
    """
    if declared != "auto":
        return declared
    return "combined_cv" if len(studies) <= 1 else "presplit"


def assert_disjoint(index_a: pd.Index, index_b: pd.Index, task_name: str) -> None:
    """
    Pooling assumes a task's train and test rows are disjoint samples; if they are not, the
    same sample would land in both a training fold and its own held-out fold. (The same
    physical sample CAN appear in both files overall -- the split was drawn per task -- which
    is why this checks only this task's filtered rows, not the whole file.)
    """
    overlap = index_a.intersection(index_b)
    if len(overlap) > 0:
        raise ValueError(f"Task '{task_name}': {len(overlap)} sample(s) appear in both splits; "
                         f"pooling would leak them across folds, e.g. {list(overlap[:5])}")


def drop_rare_classes(labels: pd.Series, min_count: int, task_name: str):
    """
    Remove only the classes stratification cannot place in every fold.

    Below min_count this is a hard constraint, not a quality judgement: StratifiedKFold only
    warns and carries on, leaving some fold with zero members of the class, which then makes
    roc_auc_ovr_weighted raise (a fold's y_true missing a class y_score has a column for) or
    silently returns fewer predict_proba columns than there are classes for a class absent
    from that fold's training set.

    :return: (filtered_labels, {class: original_count} for every class dropped)
    """
    counts = labels.value_counts()
    dropped = {str(k): int(v) for k, v in counts[counts < min_count].items()}
    if dropped:
        logger.warning("  '%s': dropping %d class(es) with < %d members, which "
                       "stratification cannot put in every fold: %s",
                       task_name, len(dropped), min_count, dropped)
    return labels[labels.isin(counts[counts >= min_count].index)], dropped


def resolve_cv_config(cv_dict) -> Dict:
    """
    Fill in cv defaults (outer_folds=5, min_samples_per_class=outer_folds) and validate.

    min_samples_per_class below outer_folds is rejected outright: a class with fewer members
    than that cannot appear in every stratified fold, which is the floor drop_rare_classes
    enforces -- see its docstring for why that's non-negotiable rather than tunable.
    """
    cv_dict = dict(cv_dict or {})
    cv_dict.setdefault("outer_folds", 5)
    cv_dict.setdefault("min_samples_per_class", cv_dict["outer_folds"])
    if cv_dict["min_samples_per_class"] < cv_dict["outer_folds"]:
        raise ValueError(
            f"cv.min_samples_per_class={cv_dict['min_samples_per_class']} is below "
            f"outer_folds={cv_dict['outer_folds']}; stratification cannot place such a class "
            f"in every fold.")
    return cv_dict
