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
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
        if (task_dir / method_name / "classification" / "cv_summary.json").exists():
            return "classification"
        if (task_dir / method_name / "regression" / "regression_scores.json").exists():
            return "regression"
        if (task_dir / method_name / "regression" / "cv_summary.json").exists():
            return "regression"
    return "unknown"


def _read_cv_summary_row(cv_path: Path, task_name: str, task_type: str) -> Dict:
    """
    A single-study task's pooled-CV result (utils/downstream_utils.py::run_single_method_cv),
    read back into the same row shape as the presplit path -- the fold mean stands in for the
    single point estimate a presplit task reports; the fold spread (std/ci_margin/values per
    fold) stays in cv_summary.json itself for anyone who wants it.
    """
    payload = json.loads(cv_path.read_text())
    aggregates = payload["aggregates"]
    status = f"OK ({payload['outer_folds']}-fold CV)"

    def mean(key: str) -> Optional[float]:
        agg = aggregates.get(key)
        return agg["mean"] if agg else None

    if task_type == "classification":
        return {
            "task": task_name, "type": "classification", "status": status,
            "accuracy": mean("accuracy"), "f1_macro": mean("f1_macro"),
            "f1_weighted": mean("f1_weighted"), "auroc": mean("auroc"),
            "mae": None, "rmse": None, "r2": None,
        }
    return {
        "task": task_name, "type": "regression", "status": status,
        "accuracy": None, "f1_macro": None, "f1_weighted": None, "auroc": None,
        "mae": mean("mae"), "rmse": mean("rmse"), "r2": mean("r2"),
    }


def _read_task_method_row(task_dir: Path, task_name: str, method_name: str, task_type: str) -> Dict:
    if task_type == "classification":
        cv_path = task_dir / method_name / "classification" / "cv_summary.json"
        if cv_path.exists():
            return _read_cv_summary_row(cv_path, task_name, task_type)

        scores_path = task_dir / method_name / "classification" / "multiclass_scores.json"
        if not scores_path.exists():
            return _missing_row(task_name, task_type)
        scores = json.loads(scores_path.read_text())
        summary, label_rows = scores[-1], scores[:-1]
        conf_mat = _parse_confusion_matrix(summary["Confusion Matrix"])
        row = {
            "task": task_name, "type": "classification", "status": "OK",
            "accuracy": summary.get("Total Accuracy"),
            "f1_macro": summary.get("Macro F1"),
            "f1_weighted": _weighted_f1_from_confusion_matrix(conf_mat),
            "auroc": _weighted_auroc(label_rows),
            "mae": None, "rmse": None, "r2": None,
        }
        # Presplit (multi-study) task: no across-fold spread exists, so
        # utils/downstream_utils.py::run_single_method attaches the inner hyperparameter
        # search's own CV std (at the winning params) to whichever metric it optimized --
        # a proxy margin, not a real held-out error bar. Not rendered as a table column
        # (same as cv_summary.json's fold spread), just exposed here for callers that want it.
        std_path = task_dir / method_name / "classification" / "search_cv_std.json"
        if std_path.exists():
            std_info = json.loads(std_path.read_text())
            row["ci_metric"] = std_info["metric"]
            row["ci_margin"] = std_info["std"]
        return row

    if task_type == "regression":
        cv_path = task_dir / method_name / "regression" / "cv_summary.json"
        if cv_path.exists():
            return _read_cv_summary_row(cv_path, task_name, task_type)

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


def _collect_run_metrics(run_output_dir: Path) -> Dict[str, Dict[str, Dict[str, Optional[float]]]]:
    """
    Same underlying scan as _collect_run_rows, reshaped for seeded aggregation.

    :return: {task_name: {method_name: {"accuracy":.., "f1_macro":.., "f1_weighted":..,
        "auroc":.., "mae":.., "rmse":.., "r2":..}}}
    """
    rows_by_method = _collect_run_rows(run_output_dir)
    metrics: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
    for method_name, rows in rows_by_method.items():
        for row in rows:
            metrics.setdefault(row["task"], {})[method_name] = {
                key: row[key]
                for key in ("accuracy", "f1_macro", "f1_weighted", "auroc", "mae", "rmse", "r2")
            }
    return metrics


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


# ==============================================================================
# ERROR BARS ACROSS REPEATED-SEED SPLITS
#
# Kept for scripts/run_raw_baselines_with_seeds.sh (which still reruns
# scripts/preprocess.py per seed to get repeated random splits) and
# scripts/aggregate_seeded_downstream_summary.py, its consumer. The three
# scripts/run_{embedding_baselines,finetune_pretrained,finetune_scratch}_with_seeds.sh
# runners no longer use this: single-study tasks now get error bars from pooled CV
# (utils/downstream_utils.py::run_task_combined_cv writes cv_summary.json, read
# transparently by _read_task_method_row above), so those three scripts run once per
# config with no seed loop, matching kd_tasks/run_downstream_final.py's one-seed-one-run
# design. This block stays for the older, separate tool that still needs it.
# ==============================================================================

def _mean_std(values: List[Optional[float]]) -> Tuple[Optional[float], Optional[float]]:
    """Sample mean/stdev of a metric across seeds, ignoring seeds with no result."""
    clean = [v for v in values if v is not None]
    if not clean:
        return None, None
    if len(clean) == 1:
        return clean[0], None
    return statistics.mean(clean), statistics.stdev(clean)


def _fmt_mean_std(mean_std: Tuple[Optional[float], Optional[float]], digits: int = 4) -> str:
    mean, std = mean_std
    if mean is None:
        return "–"
    if std is None:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


METRIC_TITLES = {
    "accuracy": "Accuracy",
    "f1_macro": "F1 (macro)",
    "f1_weighted": "F1 (weighted)",
    "auroc": "AUROC*",
    "mae": "MAE",
    "rmse": "RMSE",
    "r2": "R²",
}


def _render_error_bar_tables(rows_by_metric: Dict[str, list], methods: list, extra_column: Optional[str] = None) -> list:
    """Same layout as _render_tables, but each cell is a (mean, std) tuple rendered as 'mean ± std'."""
    lines = []
    header_cols = ([extra_column.title()] if extra_column else []) + [_display_method_name(m) for m in methods]

    for metric_key, rows in rows_by_metric.items():
        lines.append(f"## {METRIC_TITLES[metric_key]}")
        lines.append("")
        lines.append("| Task | " + " | ".join(header_cols) + " |")
        lines.append("|" + "---|" * (1 + len(header_cols)))
        for row in rows:
            cells = [row["task"]] + ([row[extra_column]] if extra_column else [])
            cells += [_fmt_mean_std(row["values"].get(m, (None, None))) for m in methods]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    return lines


def combine_seeded_downstream_summaries(seeds_root_dir: Path, output_path: Optional[Path] = None) -> None:
    """
    Aggregate downstream task results across repeated train/test-split seeds into
    mean +/- (sample) standard deviation tables.

    Expects the directory layout produced by rerunning the same eval config(s) once per
    split seed, e.g. via scripts/run_raw_baselines_with_seeds.sh:
        <seeds_root_dir>/seed_<i>/<config_name>/downstream_tasks/<task>/<method>/...

    :param seeds_root_dir: Parent directory containing one subdirectory per seed
        (named "seed_<i>"), e.g. outputs/eval/raw_baselines_seeds/.
    :param output_path: Where to write the resulting summary; defaults to
        <seeds_root_dir>/combined_summary_with_error_bars.md.
    """
    seeds_root_dir = Path(seeds_root_dir)
    output_path = Path(output_path) if output_path else seeds_root_dir / "combined_summary_with_error_bars.md"

    seed_dirs = sorted(d for d in seeds_root_dir.iterdir() if d.is_dir() and d.name.startswith("seed_"))
    if not seed_dirs:
        logger.logger.warning(f"No seed_* directories found under {seeds_root_dir}")
        return

    config_names = sorted({
        d.name for seed_dir in seed_dirs for d in seed_dir.iterdir()
        if d.is_dir() and (d / "downstream_tasks").is_dir()
    })

    lines = [
        "# Combined downstream task summary (mean ± std across seeds)",
        "",
        f"Seeds: {', '.join(d.name for d in seed_dirs)} (n={len(seed_dirs)})",
        f"Configs: {', '.join(config_names) if config_names else '(none found)'}",
        "",
        "Each cell is the sample mean ± sample standard deviation of the metric across the "
        "seeds above -- each seed reruns preprocessing with a different stratified train/test "
        "split, then trains and evaluates from scratch on that split. A cell with no `±` means "
        "only one seed produced a result for that task/method/config.",
        "",
    ]

    if not config_names:
        lines.append("(no downstream task results found on disk)")
        output_path.write_text("\n".join(lines) + "\n")
        return

    per_config_seed_metrics = {
        config_name: [_collect_run_metrics(seed_dir / config_name) for seed_dir in seed_dirs]
        for config_name in config_names
    }

    all_methods = _order_methods({
        method
        for seed_metrics_list in per_config_seed_metrics.values()
        for run_metrics in seed_metrics_list
        for per_method in run_metrics.values()
        for method in per_method
    })

    rows_by_metric = {metric_key: [] for metric_key in METRIC_TITLES}
    for config_name in config_names:
        seed_metrics_list = per_config_seed_metrics[config_name]
        all_tasks = sorted({t for m in seed_metrics_list for t in m})
        for task_name in all_tasks:
            values_by_metric = {metric_key: {} for metric_key in METRIC_TITLES}
            for method in all_methods:
                for metric_key in METRIC_TITLES:
                    vals = [(m.get(task_name, {}).get(method) or {}).get(metric_key) for m in seed_metrics_list]
                    values_by_metric[metric_key][method] = _mean_std(vals)
            for metric_key, values in values_by_metric.items():
                rows_by_metric[metric_key].append({"task": task_name, "config": config_name, "values": values})

    lines += _render_error_bar_tables(rows_by_metric, all_methods, extra_column="config")
    lines.append(AUROC_FOOTNOTE)

    output_path.write_text("\n".join(lines) + "\n")
    logger.logger.info(f"Saved seeded combined downstream task summary to {output_path}")

