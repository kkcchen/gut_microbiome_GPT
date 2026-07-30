#!/usr/bin/env python3
"""
Downstream evaluation on the original HMC tasks (downstream_train.h5ad +
downstream_test.h5ad). Replaces runners/run_downstream_from_raw.py.

What changed:

  1. Classification only. `age` and `bmi` are the only continuous tasks and
     are gone from the config, as are random forest and tabpfn -- `linear`
     and `xgboost` are the methods.

  2. How a task is evaluated depends on how many studies it spans, read off
     obs["study_id"]:

       single-study -> pool train+test and run nested CV (outer folds for the
                       estimate, a full grid search inside each), giving a
                       mean with an error margin instead of one point
                       estimate on a few dozen test rows.

       multi-study  -> keep the existing split: grid search on train, one
                       evaluation on test. On the current data that is
                       exactly `location` and `sex`, whose train and test
                       studies are disjoint -- the split is a cross-study
                       holdout worth preserving.

  3. Features come from the config: each "feature_sets" entry names a source
     (.X or an .obsm key) plus a per-sample normalization, and every task and
     method runs once per feature set. Only "raw" (untransformed counts) is
     active; add entries to compare normalizations without touching this file.

Outputs. Per task, self-contained, written as soon as that task finishes:
    output/<exp>/<task>/summary.txt    readable: every feature set x method,
                                        mean +/- 95% CI margin per metric
    output/<exp>/<task>/summary.csv    the same, tidy, one row per metric
    output/<exp>/<task>/<fs>/<method>/result.json
                                        full result incl. per-fold values and
                                        the parameters chosen in each fold

Run-wide, only written by an unfiltered run or by --collect:
    results/<exp>.csv            wide table, mean per (feature_set, method) x task
    results/<exp>_summary.csv    mean, std, 95% CI margin, class info
    results/<exp>_folds.csv      one row per outer fold per metric

Usage:
    python runners/run_downstream_final.py
    python runners/run_downstream_final.py --tasks sex crohns_disease --dry-run

One task per Slurm array job, then collect. Every job writes only its own
task directory, so nothing races; --collect merges them afterwards:
    python runners/run_downstream_final.py --tasks $TASK
    python runners/run_downstream_final.py --collect
"""

import argparse
import json
import os
import sys

import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from kd_tasks.utils import seed_everything  # noqa: E402
from kd_tasks.downstream_final_utils import (  # noqa: E402
    ExecConfig, ResultsTable, ResultsWriter, build_task_split, class_feasibility,
    collect_results, combine_splits, concat_features, cv_for, fit_count, load_cached,
    load_config, load_features, logger, preprocess_features, read_obs, record_result,
    resolve_split_mode, run_nested_cv, run_presplit, save_cached, setup_logging,
    write_task_summary,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "downstream_final.json"))
    p.add_argument("--tasks", nargs="*", help="only these tasks (default: all in config)")
    p.add_argument("--feature-sets", nargs="*", help="only these feature sets")
    p.add_argument("--dry-run", action="store_true", help="print the plan, then exit")
    p.add_argument("--no-cache", action="store_true", help="recompute finished work")
    p.add_argument("--collect", action="store_true",
                   help="rebuild the run-wide tables from finished per-task results, "
                        "without fitting anything (run this after a job array)")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def print_plan(config, tasks, feature_sets, obs_frames):
    """
    What will run and what it costs, before anything expensive starts. Nested
    CV multiplies fast -- the xgboost grid is 72 candidates, so one task costs
    outer * (72 * inner + 1) fits -- and skipped tasks are much better spotted
    here than as a gap in the results table.
    """
    total = 0
    logger.info("")
    logger.info("%-40s %-11s %-9s %6s %8s %9s",
                "task", "mode", "studies", "n", "classes", "fits/fset")
    logger.info("-" * 90)

    for name, task_conf in tasks.items():
        task = build_task_split(name, obs_frames, task_conf["ignored_labels"])
        if task.label_col != "categorical_label":
            logger.info("%-40s %-11s %s", name, "SKIP", "continuous label (no regression)")
            continue

        cv = cv_for(config, task_conf)
        mode = resolve_split_mode(task, task_conf["split_mode"])
        n = sum(len(ids) for ids in task.sample_ids.values())

        # Judge feasibility on whatever the mode actually trains from.
        judged = (pd.concat(list(task.labels.values())) if mode == "combined_cv"
                  else task.labels["train"])
        usable, n_classes, runnable = class_feasibility(judged, cv["min_samples_per_class"])

        if not runnable:
            logger.info("%-40s %-11s %-9s %6d %8s", name, "SKIP", "", n,
                        f"{usable}/{n_classes}")
            logger.warning("  '%s' will be skipped: only %d class(es) reach "
                           "min_samples_per_class=%d. Lower it for this task to run anyway.",
                           name, usable, cv["min_samples_per_class"])
            continue

        fits = fit_count(cv, task_conf["methods"], mode)
        total += fits * len(feature_sets)
        logger.info("%-40s %-11s %-9d %6d %8s %9d", name, mode, len(task.studies), n,
                    f"{usable}/{n_classes}", fits)

    logger.info("-" * 90)
    logger.info("%d feature set(s) -> ~%d model fits total", len(feature_sets), total)
    logger.info("")


def main():
    args = parse_args()
    setup_logging(args.verbose)

    config = load_config(args.config)
    seed = config["seed"]
    seed_everything(seed)

    dataset = config["dataset"]
    train_h5ad, test_h5ad = dataset["train_h5ad"], dataset["test_h5ad"]
    output_root = dataset["output_dir"]
    experiment = dataset["experiment_name"]

    if args.collect:
        collect_results(output_root, experiment, dataset["results_dir"])
        return

    tasks = config["tasks"]
    if args.tasks:
        tasks = {k: v for k, v in tasks.items() if k in args.tasks}
    feature_sets = config["feature_sets"]
    if args.feature_sets:
        feature_sets = {k: v for k, v in feature_sets.items() if k in args.feature_sets}

    logger.info("Config: %s | feature sets: %s", args.config, list(feature_sets))
    obs_frames = {"train": read_obs(train_h5ad), "test": read_obs(test_h5ad)}

    print_plan(config, tasks, feature_sets, obs_frames)
    if args.dry_run:
        return

    exec_conf = ExecConfig.build(config["execution"]["device"], config["execution"]["n_cpus"])
    logger.info("Execution: %s", exec_conf.describe())

    # Load every feature set once up front (~64 MB each), so the loop below can
    # run task-outer and finish a task completely before moving on. That is
    # what lets one Slurm array job own one task.
    features = {}
    for fs_name, spec in feature_sets.items():
        logger.info("Preparing feature set '%s'", fs_name)
        # Load both splits raw, then threshold and normalize the combined
        # matrix -- thresholds are computed over train and test together, as
        # in the previous normalization pipeline.
        features[fs_name] = preprocess_features(
            concat_features(load_features(train_h5ad, spec, obs_frames["train"].index),
                            load_features(test_h5ad, spec, obs_frames["test"].index)),
            spec)

    results = ResultsTable(experiment, row_key_names=("feature_set", "method"),
                           results_root=dataset["results_dir"])
    writer = ResultsWriter(experiment, results_root=dataset["results_dir"])

    for name, task_conf in tasks.items():
        task = build_task_split(name, obs_frames, task_conf["ignored_labels"])
        if task.label_col != "categorical_label":
            continue

        cv = cv_for(config, task_conf)
        mode = resolve_split_mode(task, task_conf["split_mode"])
        task_dir = os.path.join(output_root, name)
        logger.info("")
        logger.info("=== task: %s (studies=%s, mode=%s) ===", name, task.studies, mode)

        for fs_name, spec in feature_sets.items():
            for method, method_conf in task_conf["methods"].items():
                out_dir = os.path.join(task_dir, fs_name, method)
                os.makedirs(out_dir, exist_ok=True)
                cache_path = os.path.join(out_dir, "result.json")

                fingerprint = json.dumps(
                    {"task": name, "feature_set": fs_name, "method": method, "spec": spec,
                     "method_conf": method_conf, "cv": cv, "seed": seed, "mode": mode,
                     "ignored": sorted(task_conf["ignored_labels"])},
                    sort_keys=True, default=str)

                payload = None if args.no_cache else load_cached(cache_path, fingerprint)
                if payload is not None:
                    logger.info("  [%s] %s: cached", fs_name, method)
                else:
                    logger.info("  [%s] %s", fs_name, method)
                    if mode == "combined_cv":
                        payload = run_nested_cv(method, method_conf, features[fs_name],
                                                combine_splits(task),
                                                cv, exec_conf, seed, out_dir, name)
                    else:
                        payload = run_presplit(method, method_conf, features[fs_name],
                                               task.labels["train"], task.labels["test"],
                                               cv, exec_conf, seed, out_dir, name)
                    if payload is None:
                        continue
                    save_cached(cache_path, fingerprint, payload)

                record_result(results, writer, fs_name, method, name, payload)

        # Reads every result.json under the task, so a job that ran only some
        # feature sets still leaves a summary covering all finished work.
        write_task_summary(task_dir, name)

    # Under a job array every job would race for the same run-wide paths, so
    # a filtered run only leaves its per-task summaries behind.
    if args.tasks or args.feature_sets:
        logger.info("")
        logger.info("Partial run: per-task summaries written under %s.", output_root)
        logger.info("Run with --collect once all jobs finish to build the run-wide tables.")
    else:
        results.save()
        writer.save()


if __name__ == "__main__":
    main()
