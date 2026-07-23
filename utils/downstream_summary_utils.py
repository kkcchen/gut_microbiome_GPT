"""
Compiles per-(task, method) downstream evaluation results (scattered across
outputs/eval/embedding_baselines/<config_name>/downstream_tasks/<task>/<method>/...)
into Markdown summary tables, one table per method, in the same row shape as
utils/finetune_orchestration.py::write_finetune_summary
(Task | Type | Status | Accuracy | F1 (macro) | F1 (weighted) | AUROC* | MAE | RMSE | R²).

Pure disk-scanning (no cfg, no in-memory results) so these can be re-run
standalone against any existing output directory, and so a per-config
summary.md stays accurate even if the eval run is killed partway through.

F1 (weighted) and AUROC aren't stored directly by downstream_eval_utils.py, so
they're derived from what IS stored: F1 (weighted) from the saved confusion
matrix (support-weighted per-class F1, identical to sklearn's
f1_score(average="weighted")), and AUROC from the saved per-label AUCs
weighted by each label's n_samples (identical to sklearn's
roc_auc_score(average="weighted") for one-vs-rest multiclass).
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from trainers import logger

METHOD_DISPLAY_NAMES = {
    "xgboost": "XGBoost",
    "random_forest": "Random Forest",
    "linear": "Linear",
    "tabpfn": "TabPFN",
}
METHOD_ORDER = ["xgboost", "random_forest", "linear", "tabpfn"]

AUROC_FOOTNOTE = (
    "*AUROC is the sample-weighted average of the per-class one-vs-rest AUCs saved for "
    'each task (equivalent to sklearn\'s `average="weighted"`); F1 (weighted) is likewise '
    "derived from the saved confusion matrix (support-weighted average across classes)."
)


def _display_method_name(method_name: str) -> str:
    return METHOD_DISPLAY_NAMES.get(method_name, method_name.replace("_", " ").title())


def _order_methods(method_names) -> List[str]:
    known = [m for m in METHOD_ORDER if m in method_names]
    unknown = sorted(m for m in method_names if m not in METHOD_ORDER)
    return known + unknown


def _fmt(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "–"
    return f"{value:.{digits}f}"


def _parse_confusion_matrix(conf_mat_rows: List[str]) -> np.ndarray:
    return np.array([[int(x) for x in re.findall(r"-?\d+", row)] for row in conf_mat_rows])


def _weighted_f1_from_confusion_matrix(conf_mat: np.ndarray) -> Optional[float]:
    support = conf_mat.sum(axis=1)
    if support.sum() == 0:
        return None
    f1s = []
    for c in range(conf_mat.shape[0]):
        tp = conf_mat[c, c]
        fp = conf_mat[:, c].sum() - tp
        fn = conf_mat[c, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0)
    return float(np.average(f1s, weights=support))


def _weighted_auroc(label_rows: List[Dict]) -> Optional[float]:
    aucs = [row["AUC (ROC)"] for row in label_rows]
    weights = [row["n_samples"] for row in label_rows]
    if not aucs or sum(weights) == 0:
        return None
    return float(np.average(aucs, weights=weights))


def _missing_row(task_name: str, task_type: str) -> Dict:
    return {
        "task": task_name, "type": task_type, "status": "MISSING",
        "accuracy": None, "f1_macro": None, "f1_weighted": None, "auroc": None,
        "mae": None, "rmse": None, "r2": None,
    }


def _determine_task_type(task_dir: Path, method_names: List[str]) -> str:
    for method_name in method_names:
        if (task_dir / method_name / "classification" / "multiclass_scores.json").exists():
            return "classification"
        if (task_dir / method_name / "regression" / "regression_scores.json").exists():
            return "regression"
    return "unknown"


def _read_task_method_row(task_dir: Path, task_name: str, method_name: str, task_type: str) -> Dict:
    if task_type == "classification":
        scores_path = task_dir / method_name / "classification" / "multiclass_scores.json"
        if not scores_path.exists():
            return _missing_row(task_name, task_type)
        scores = json.loads(scores_path.read_text())
        summary, label_rows = scores[-1], scores[:-1]
        conf_mat = _parse_confusion_matrix(summary["Confusion Matrix"])
        return {
            "task": task_name, "type": "classification", "status": "OK",
            "accuracy": summary.get("Total Accuracy"),
            "f1_macro": summary.get("Macro F1"),
            "f1_weighted": _weighted_f1_from_confusion_matrix(conf_mat),
            "auroc": _weighted_auroc(label_rows),
            "mae": None, "rmse": None, "r2": None,
        }

    if task_type == "regression":
        scores_path = task_dir / method_name / "regression" / "regression_scores.json"
        if not scores_path.exists():
            return _missing_row(task_name, task_type)
        scores = json.loads(scores_path.read_text())
        mse = scores.get("Mean Squared Error")
        return {
            "task": task_name, "type": "regression", "status": "OK",
            "accuracy": None, "f1_macro": None, "f1_weighted": None, "auroc": None,
            "mae": scores.get("Mean Absolute Error"),
            "rmse": mse ** 0.5 if mse is not None else None,
            "r2": scores.get("R-squared"),
        }

    return _missing_row(task_name, task_type)


def _collect_run_rows(run_output_dir: Path) -> Dict[str, List[Dict]]:
    """
    :return: {method_name: [one row dict per task, sorted by task name]}, covering
        the union of methods found under any task dir (missing task/method
        combos are filled in as "MISSING" rows rather than omitted).
    """
    downstream_tasks_dir = run_output_dir / "downstream_tasks"
    if not downstream_tasks_dir.is_dir():
        return {}

    task_dirs = sorted(d for d in downstream_tasks_dir.iterdir() if d.is_dir())
    all_methods = _order_methods({
        d.name for task_dir in task_dirs for d in task_dir.iterdir() if d.is_dir()
    })

    rows_by_method: Dict[str, List[Dict]] = {m: [] for m in all_methods}
    for task_dir in task_dirs:
        method_names = [d.name for d in task_dir.iterdir() if d.is_dir()]
        task_type = _determine_task_type(task_dir, method_names)
        for method_name in all_methods:
            if method_name in method_names:
                row = _read_task_method_row(task_dir, task_dir.name, method_name, task_type)
            else:
                row = _missing_row(task_dir.name, task_type)
            rows_by_method[method_name].append(row)

    return rows_by_method


def _table_header(extra_column: Optional[str]) -> List[str]:
    cols = ["Task"] + ([extra_column] if extra_column else [])
    cols += ["Type", "Status", "Accuracy", "F1 (macro)", "F1 (weighted)", "AUROC*", "MAE", "RMSE", "R²"]
    return ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]


def _table_row(row: Dict, extra_value: Optional[str]) -> str:
    cells = [row["task"]] + ([extra_value] if extra_value is not None else [])
    cells += [
        row["type"], row["status"],
        _fmt(row["accuracy"]), _fmt(row["f1_macro"]), _fmt(row["f1_weighted"]), _fmt(row["auroc"]),
        _fmt(row["mae"]), _fmt(row["rmse"]), _fmt(row["r2"]),
    ]
    return "| " + " | ".join(cells) + " |"


def write_downstream_summary(run_output_dir: Path) -> None:
    """
    Write <run_output_dir>/summary.md: one Markdown table per method (Task |
    Type | Status | Accuracy | F1 (macro) | F1 (weighted) | AUROC* | MAE |
    RMSE | R²). Then refreshes the combined summary across all sibling configs.

    :param run_output_dir: A single config's eval output dir, e.g.
        outputs/eval/embedding_baselines/pretrained_baseline_bins20_mask030.
    """
    run_output_dir = Path(run_output_dir)
    rows_by_method = _collect_run_rows(run_output_dir)

    lines = [
        f"# Downstream task summary — {run_output_dir.name}", "",
        f"Source: `{run_output_dir / 'downstream_tasks'}`", "",
    ]
    if rows_by_method:
        for method_name, rows in rows_by_method.items():
            lines.append(f"## {_display_method_name(method_name)}")
            lines.append("")
            lines += _table_header(None)
            lines += [_table_row(row, None) for row in rows]
            lines.append("")
    else:
        lines += ["(no downstream task results found on disk)", ""]
    lines.append(AUROC_FOOTNOTE)

    summary_path = run_output_dir / "summary.md"
    summary_path.write_text("\n".join(lines) + "\n")
    logger.logger.info(f"Saved downstream task summary to {summary_path}")

    combine_downstream_summaries(run_output_dir.parent)


def combine_downstream_summaries(parent_dir: Path) -> None:
    """
    Aggregate every sibling config's downstream_tasks/ results directly under
    parent_dir (e.g. outputs/eval/embedding_baselines/) into one combined
    Markdown table per method at <parent_dir>/combined_summary.md, one row
    per (config, task).

    :param parent_dir: Directory containing one subdirectory per eval config
        run, e.g. outputs/eval/embedding_baselines/.
    """
    parent_dir = Path(parent_dir)
    run_dirs = sorted(
        d for d in parent_dir.iterdir()
        if d.is_dir() and (d / "downstream_tasks").is_dir()
    )

    per_run_rows = {d.name: _collect_run_rows(d) for d in run_dirs}
    all_methods = _order_methods({m for rows in per_run_rows.values() for m in rows})

    lines = [
        "# Combined downstream task summary", "",
        f"Configs: {', '.join(d.name for d in run_dirs) if run_dirs else '(none found)'}", "",
    ]
    if all_methods:
        for method_name in all_methods:
            lines.append(f"## {_display_method_name(method_name)}")
            lines.append("")
            lines += _table_header("Config")
            for run_name, rows_by_method in per_run_rows.items():
                for row in rows_by_method.get(method_name, []):
                    lines.append(_table_row(row, run_name))
            lines.append("")
    else:
        lines += ["(no downstream task results found on disk)", ""]
    lines.append(AUROC_FOOTNOTE)

    combined_path = parent_dir / "combined_summary.md"
    combined_path.write_text("\n".join(lines) + "\n")
    logger.logger.info(f"Saved combined downstream task summary to {combined_path}")
