"""
Compiles per-(task, method) downstream evaluation results (scattered across
outputs/eval/embedding_baselines/<config_name>/downstream_tasks/<task>/<method>/...)
into Markdown summary tables: task rows x method columns.

Pure disk-scanning (no cfg, no in-memory results) so these can be re-run
standalone against any existing output directory, and so a per-config
summary.md stays accurate even if the eval run is killed partway through.
"""
import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from trainers import logger

METHOD_DISPLAY_NAMES = {
    "xgboost": "XGBoost",
    "random_forest": "Random Forest",
    "linear": "Linear",
    "tabpfn": "TabPFN",
}
METHOD_ORDER = ["xgboost", "random_forest", "linear", "tabpfn"]


def _display_method_name(method_name: str) -> str:
    return METHOD_DISPLAY_NAMES.get(method_name, method_name.replace("_", " ").title())


def _order_methods(method_names) -> list:
    known = [m for m in METHOD_ORDER if m in method_names]
    unknown = sorted(m for m in method_names if m not in METHOD_ORDER)
    return known + unknown


def _fmt(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "–"
    return f"{value:.{digits}f}"


def _read_task_method_metrics(task_dir: Path, method_name: str) -> Optional[Dict[str, float]]:
    """
    Read Accuracy/Macro F1 for one (task, method) from disk, or None if the
    method wasn't run, failed, or the task is a regression task (accuracy/F1
    don't apply -- regression tasks are excluded from these tables).
    """
    classification_scores = task_dir / method_name / "classification" / "multiclass_scores.json"
    if classification_scores.exists():
        scores = json.loads(classification_scores.read_text())
        summary = scores[-1]
        return {
            "accuracy": summary.get("Total Accuracy"),
            "macro_f1": summary.get("Macro F1"),
        }

    regression_scores = task_dir / method_name / "regression" / "regression_scores.json"
    if regression_scores.exists():
        return None

    return None


def _collect_run_metrics(run_output_dir: Path) -> Dict[str, Dict[str, Optional[Dict[str, float]]]]:
    """
    Scan <run_output_dir>/downstream_tasks/*/*/ for every task/method actually
    present on disk.

    :return: {task_name: {method_name: {"accuracy":.., "macro_f1":..} or None}},
        with regression tasks (no classification/ subdir under any method)
        omitted entirely.
    """
    downstream_tasks_dir = run_output_dir / "downstream_tasks"
    if not downstream_tasks_dir.is_dir():
        return {}

    results: Dict[str, Dict[str, Optional[Dict[str, float]]]] = {}
    for task_dir in sorted(d for d in downstream_tasks_dir.iterdir() if d.is_dir()):
        method_names = sorted(d.name for d in task_dir.iterdir() if d.is_dir())
        is_regression_task = all(
            (task_dir / m / "regression" / "regression_scores.json").exists()
            for m in method_names
        ) if method_names else False
        if is_regression_task:
            continue

        results[task_dir.name] = {
            method_name: _read_task_method_metrics(task_dir, method_name)
            for method_name in method_names
        }

    return results


def _render_tables(rows_by_metric: Dict[str, list], methods: list, extra_column: Optional[str] = None) -> list:
    """Build Markdown lines for one table per metric in rows_by_metric."""
    lines = []
    metric_titles = {"accuracy": "Accuracy", "macro_f1": "Macro F1"}
    header_cols = ([extra_column.title()] if extra_column else []) + [_display_method_name(m) for m in methods]

    for metric_key, rows in rows_by_metric.items():
        lines.append(f"## {metric_titles[metric_key]}")
        lines.append("")
        lines.append("| Task | " + " | ".join(header_cols) + " |")
        lines.append("|" + "---|" * (1 + len(header_cols)))
        for row in rows:
            cells = [row["task"]] + ([row[extra_column]] if extra_column else [])
            cells += [_fmt(row["values"].get(m)) for m in methods]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    return lines


def write_downstream_summary(run_output_dir: Path) -> None:
    """
    Write <run_output_dir>/summary.md: task rows x method columns, one table
    each for Accuracy and Macro F1. Then refreshes the combined summary across
    all sibling configs.

    :param run_output_dir: A single config's eval output dir, e.g.
        outputs/eval/embedding_baselines/pretrained_baseline_bins20_mask030.
    """
    run_output_dir = Path(run_output_dir)
    task_metrics = _collect_run_metrics(run_output_dir)

    methods = _order_methods({m for per_method in task_metrics.values() for m in per_method})

    rows_by_metric = {"accuracy": [], "macro_f1": []}
    for task_name, per_method in task_metrics.items():
        values_acc = {m: (per_method.get(m) or {}).get("accuracy") for m in methods}
        values_f1 = {m: (per_method.get(m) or {}).get("macro_f1") for m in methods}
        rows_by_metric["accuracy"].append({"task": task_name, "values": values_acc})
        rows_by_metric["macro_f1"].append({"task": task_name, "values": values_f1})

    lines = [f"# Downstream task summary — {run_output_dir.name}", "", f"Source: `{run_output_dir / 'downstream_tasks'}`", ""]
    if task_metrics:
        lines += _render_tables(rows_by_metric, methods)
    else:
        lines.append("(no downstream task results found on disk)")
        lines.append("")

    summary_path = run_output_dir / "summary.md"
    summary_path.write_text("\n".join(lines) + "\n")
    logger.logger.info(f"Saved downstream task summary to {summary_path}")

    combine_downstream_summaries(run_output_dir.parent)


def combine_downstream_summaries(parent_dir: Path) -> None:
    """
    Aggregate every sibling config's downstream_tasks/ results directly under
    parent_dir (e.g. outputs/eval/embedding_baselines/) into one combined
    Markdown table at <parent_dir>/combined_summary.md, one row per
    (task, config).

    :param parent_dir: Directory containing one subdirectory per eval config
        run, e.g. outputs/eval/embedding_baselines/.
    """
    parent_dir = Path(parent_dir)
    run_dirs = sorted(
        d for d in parent_dir.iterdir()
        if d.is_dir() and (d / "downstream_tasks").is_dir()
    )

    per_run_metrics = {d.name: _collect_run_metrics(d) for d in run_dirs}
    all_methods = _order_methods({
        m for run_metrics in per_run_metrics.values()
        for per_method in run_metrics.values()
        for m in per_method
    })
    all_tasks = sorted({t for run_metrics in per_run_metrics.values() for t in run_metrics})

    rows_by_metric = {"accuracy": [], "macro_f1": []}
    for task_name in all_tasks:
        for run_name in per_run_metrics:
            per_method = per_run_metrics[run_name].get(task_name, {})
            values_acc = {m: (per_method.get(m) or {}).get("accuracy") for m in all_methods}
            values_f1 = {m: (per_method.get(m) or {}).get("macro_f1") for m in all_methods}
            rows_by_metric["accuracy"].append({"task": task_name, "config": run_name, "values": values_acc})
            rows_by_metric["macro_f1"].append({"task": task_name, "config": run_name, "values": values_f1})

    lines = [
        "# Combined downstream task summary",
        "",
        f"Configs: {', '.join(d.name for d in run_dirs) if run_dirs else '(none found)'}",
        "",
    ]
    if all_tasks:
        lines += _render_tables(rows_by_metric, all_methods, extra_column="config")
    else:
        lines.append("(no downstream task results found on disk)")
        lines.append("")

    combined_path = parent_dir / "combined_summary.md"
    combined_path.write_text("\n".join(lines) + "\n")
    logger.logger.info(f"Saved combined downstream task summary to {combined_path}")


# ==============================================================================
# ERROR BARS ACROSS REPEATED-SEED SPLITS
#
# There is no k-fold CV at evaluation time in this pipeline (the cv= in
# GridSearchCV/RandomizedSearchCV only picks hyperparameters internally; the
# reported metrics come from one fixed train/test split). To get error bars,
# rerun preprocessing + eval with several different split seeds and aggregate
# here across the resulting per-seed result directories.
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


def _render_error_bar_tables(rows_by_metric: Dict[str, list], methods: list, extra_column: Optional[str] = None) -> list:
    """Same layout as _render_tables, but each cell is a (mean, std) tuple rendered as 'mean ± std'."""
    lines = []
    metric_titles = {"accuracy": "Accuracy", "macro_f1": "Macro F1"}
    header_cols = ([extra_column.title()] if extra_column else []) + [_display_method_name(m) for m in methods]

    for metric_key, rows in rows_by_metric.items():
        lines.append(f"## {metric_titles[metric_key]}")
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

    rows_by_metric = {"accuracy": [], "macro_f1": []}
    for config_name in config_names:
        seed_metrics_list = per_config_seed_metrics[config_name]
        all_tasks = sorted({t for m in seed_metrics_list for t in m})
        for task_name in all_tasks:
            values_acc, values_f1 = {}, {}
            for method in all_methods:
                acc_vals = [(m.get(task_name, {}).get(method) or {}).get("accuracy") for m in seed_metrics_list]
                f1_vals = [(m.get(task_name, {}).get(method) or {}).get("macro_f1") for m in seed_metrics_list]
                values_acc[method] = _mean_std(acc_vals)
                values_f1[method] = _mean_std(f1_vals)
            rows_by_metric["accuracy"].append({"task": task_name, "config": config_name, "values": values_acc})
            rows_by_metric["macro_f1"].append({"task": task_name, "config": config_name, "values": values_f1})

    lines += _render_error_bar_tables(rows_by_metric, all_methods, extra_column="config")

    output_path.write_text("\n".join(lines) + "\n")
    logger.logger.info(f"Saved seeded combined downstream task summary to {output_path}")
