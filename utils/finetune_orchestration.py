"""
Automatic pretrain -> finetune-all-tasks orchestration.

After a pretraining run (scripts/train.py) finishes, this module drives one
finetuning run per task in configs/finetune/task_registry.yaml against the
checkpoint that pretraining just produced, reusing the existing
scripts/finetune.py pipeline (data prep, class weights, training loop, and
per-task test_metrics.yaml) unchanged. It then collects all the per-task
test_metrics.yaml files into a single human-readable Markdown summary.
"""
import copy
import os
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anndata as ad
from omegaconf import OmegaConf

from trainers import logger
from .downstream_split_mode import resolve_cv_config
from .downstream_summary_utils import METRIC_TITLES, _mean_std, _render_error_bar_tables
from .finetune_cv_utils import resolve_task_mode, run_task_cv

TASK_REGISTRY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs", "finetune", "task_registry.yaml",
)
FINETUNE_TEMPLATE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs", "finetune", "default.yaml",
)

# Describes trainers/trainer.py::evaluate_on_test_set's own AUROC computation -- distinct
# from utils/downstream_summary_utils.py's AUROC_FOOTNOTE, which describes the classical-ML
# embedding-baselines pipeline's per-label-AUC-reweighting instead.
FINETUNE_AUROC_FOOTNOTE = (
    "*AUROC is the class-weighted (support-weighted) macro-average across classes "
    'for multi-class tasks (sklearn `average="weighted"`); the standard AUROC for binary tasks.'
)


def load_task_registry(path: str = TASK_REGISTRY_PATH) -> Dict[str, Dict]:
    """Load the {task_name: {finetune_task, label_column, ignored_labels}} registry."""
    registry = OmegaConf.load(path)
    return OmegaConf.to_container(registry.tasks, resolve=True)


def require_finetune_section(pretrain_cfg) -> None:
    """Fail fast (before any expensive work) if the pretrain config can't drive auto-finetuning."""
    if "finetune" not in pretrain_cfg:
        raise ValueError(
            "Pretrain config is missing a 'finetune' section, required to drive the "
            "automatic finetune-all-tasks chain. Add a 'finetune:' block (paths.downstream_train, "
            "paths.downstream_test, training hyperparameters) to the pretrain config."
        )


def require_single_process(accelerator) -> None:
    """
    Fail fast (before any expensive work, on every rank) if pretraining was launched
    multi-GPU/multi-process.

    The auto-finetune chain reuses the pretraining process's Accelerator: it only
    runs on the main process (other ranks just wait), and each of the 12 finetune
    sub-runs constructs its own Accelerator in the same process. Accelerate's
    AcceleratorState is a process-wide singleton, so under num_processes>1 this
    deadlocks (the finetune sub-run's DDP wrapping expects collective ops from
    ranks that never participate) or crashes (destroy_process_group() from the
    first sub-run's cleanup breaks the next one). Checked unconditionally (not
    gated by is_main_process) so every rank fails together instead of the other
    ranks hanging forever at a barrier the main process never reaches.
    """
    if accelerator.num_processes > 1:
        raise RuntimeError(
            f"This pretrain config would run with {accelerator.num_processes} processes, but the "
            "automatic finetune-all-tasks chain (utils/finetune_orchestration.py) only supports "
            "single-GPU/single-process pretraining -- it deadlocks under multi-GPU DDP. Launch this "
            "config on a single GPU/process instead (e.g. `--gres=gpu:1` with no `--multi_gpu` / "
            "`--num_processes>1` accelerate launch flags)."
        )


def build_finetune_config(
    pretrain_cfg,
    pretrain_output_dir: str,
    task_name: str,
    task_spec: Dict,
    finetune_output_root: Optional[str] = None,
    fold_paths: Optional[Tuple[str, str]] = None,
    fold_idx: Optional[int] = None,
):
    """
    Assemble a complete finetuning config for one downstream task, based on the
    existing configs/finetune/default.yaml template (so every key the shared
    finetuning pipeline expects is already present and valid), with:
      - paths pointed at the checkpoint/vocab this pretraining run just produced,
      - model.params copied from the pretrain config so the finetuned model always
        matches the architecture that was actually pretrained,
      - data settings copied from the pretrain config so normalization/sequence
        handling matches what the model was pretrained on,
      - base finetuning hyperparameters taken from pretrain_cfg.finetune,
      - the task's own finetune_task/label_column from the registry.

    :param pretrain_cfg: OmegaConf config of the pretraining run.
    :param pretrain_output_dir: Output directory of the completed pretraining run.
    :param task_name: Downstream task name (matches adata.obs['downstream_task']).
    :param task_spec: Registry entry for this task ({finetune_task, label_column}).
    :param finetune_output_root: If set, finetune results (task_output_dir, and therefore
        model_config_path/best_model/test_metrics.yaml) are written under here instead of
        under pretrain_output_dir -- the checkpoint/vocab are still read from
        pretrain_output_dir either way. Used to re-finetune an already-trained checkpoint
        against a different data split without touching or overwriting that checkpoint's
        own results.
    :param fold_paths: (downstream_train, downstream_test) h5ad paths for one outer CV fold
        of a single-study task (see utils/finetune_cv_utils.py::run_task_cv). If set, these
        replace ft_cfg.paths.downstream_train/downstream_test; taxa_vocab_path/
        batch_vocab_path/checkpoint_path are unaffected either way -- they're pretraining
        artifacts, not fold-specific.
    :param fold_idx: Outer fold index, used only to give this fold's task_output_dir its own
        fold_<i> subdirectory so folds never overwrite each other's checkpoints/results.
    :return: OmegaConf config ready to pass to scripts.finetune.main().
    """
    require_finetune_section(pretrain_cfg)
    cfg = OmegaConf.load(FINETUNE_TEMPLATE_PATH)
    ft_cfg = pretrain_cfg.finetune

    result_root = finetune_output_root if finetune_output_root is not None else pretrain_output_dir
    task_output_dir = os.path.join(result_root, "finetune", task_name)
    if fold_idx is not None:
        task_output_dir = os.path.join(task_output_dir, f"fold_{fold_idx}")

    if fold_paths is not None:
        cfg.paths.downstream_train, cfg.paths.downstream_test = fold_paths
    else:
        cfg.paths.downstream_train = ft_cfg.paths.downstream_train
        cfg.paths.downstream_test = ft_cfg.paths.get("downstream_test", None)
    cfg.paths.output_dir = task_output_dir
    cfg.paths.model_config_path = os.path.join(task_output_dir, "model_config.json")
    cfg.paths.taxa_vocab_path = os.path.join(pretrain_output_dir, "taxa_vocab.pkl")
    cfg.paths.batch_vocab_path = os.path.join(pretrain_output_dir, "batch_vocab.pkl")
    cfg.paths.checkpoint_path = os.path.join(
        pretrain_output_dir, "best_model", "best_model", "pytorch_model.bin"
    )

    # Disabled by default: a wandb run per task per pretrain config would multiply
    # runs 12x; enable manually on the template if per-task tracking is wanted.
    cfg.wandb.enabled = False

    cfg.data.norm_strategy = pretrain_cfg.data.norm_strategy
    cfg.data.use_batch_labels = pretrain_cfg.data.use_batch_labels
    cfg.data.max_seq_len = pretrain_cfg.data.get("max_seq_len", 200)
    cfg.data.num_bins = pretrain_cfg.data.get("num_bins", 15)
    cfg.data.num_workers = pretrain_cfg.data.get("num_workers", 4)
    cfg.data.val_size = ft_cfg.get("val_size", 0.2)
    cfg.data.finetune_task_name = task_name
    cfg.data.label_column = task_spec["label_column"]
    cfg.data.ignored_labels = task_spec.get("ignored_labels", [])

    for key, value in ft_cfg.get("training", {}).items():
        cfg.training[key] = value
    cfg.training.finetune_task = task_spec["finetune_task"]
    # Must match the pretraining run's own enable_fp16, never the template's. Accelerate's
    # AcceleratorState is a process-wide singleton shared with the pretraining Accelerator that's
    # still alive when this sub-run's Accelerator gets constructed: a mismatched mixed_precision
    # raises ValueError deep inside scripts.finetune.main(), after pretraining already finished.
    cfg.training.enable_fp16 = pretrain_cfg.training.enable_fp16

    cfg.model.params = copy.deepcopy(pretrain_cfg.model.params)

    return cfg


def run_all_downstream_finetunes(pretrain_cfg, accelerator, finetune_output_root: Optional[str] = None,
                                 mode_filter: Optional[str] = None,
                                 tasks: Optional[List[str]] = None,
                                 only_fold: Optional[int] = None) -> Dict[str, Dict]:
    """
    Finetune the just-completed pretraining run on every task in the task
    registry, writing (and re-writing, after every task) an aggregated summary
    so a hard kill partway through still leaves an accurate summary of whatever
    completed. No-ops on non-main processes.

    :param pretrain_cfg: OmegaConf config of the pretraining run that just finished.
    :param accelerator: Accelerator used for pretraining (checked for is_main_process).
    :param finetune_output_root: If set, results (finetune_summary.md and each task's
        finetune/<task>/) are written under here instead of under pretrain_cfg.paths.output_dir
        -- the checkpoint/vocab are still read from pretrain_cfg.paths.output_dir either way.
        Used to re-finetune an already-trained checkpoint against a different data split
        (e.g. scripts/run_finetune_pretrained_with_seeds.sh looping over split seeds) without
        overwriting that checkpoint's own results.
    :param mode_filter: If "combined_cv" or "presplit", only tasks whose auto-detected split
        mode (utils/downstream_split_mode.py::resolve_split_mode) matches are run; the rest
        are recorded with status "skipped" without doing any of the expensive pooled-CV or
        finetuning work. Used to repeat only the multi-study/presplit tasks across several
        training seeds (single-study tasks already get a real error bar from pooled CV, so
        rerunning them per seed would just waste compute) -- see
        scripts/run_finetune_pretrained_with_seeds.sh / run_finetune_scratch_with_seeds.sh.
    :param tasks: If set, only tasks whose name is in this list are run; the rest are recorded
        as "skipped", same as mode_filter. Needed because the task registry and
        finetune.paths.downstream_train/downstream_test are both global -- when a task
        registry holds tasks that live in different base h5ads (e.g. the original HMC tasks
        vs a separate dataset's tasks), one invocation's downstream_train/downstream_test
        override only applies to one dataset's tasks, so the other dataset's tasks must be
        filtered out by name rather than accidentally pointed at the wrong data.
    :param only_fold: If set, every combined_cv task trains (or skips, if already cached)
        only this one outer fold and no cv_summary.yaml is written yet -- see
        utils/finetune_cv_utils.py::run_task_cv. Lets a task's outer folds run as separate,
        parallel jobs instead of one job finetuning through all of them in sequence, which
        matters far more here than in the classical-ML pipelines since each fold is a full
        finetuning run. presplit tasks have no folds and are skipped when this is set; run
        again without only_fold once every fold of every combined_cv task is done, to
        assemble the cv_summary.yaml files (cache-hits every fold, so that pass is fast).
    :return: {task_name: {status, ...}} -- status is "ok", "failed", "skipped" (filtered out
        by mode_filter/tasks/only_fold), or "pending" (if this returned early on a non-main
        process, before anything ran). Callers can inspect this to decide whether to exit
        non-zero on partial failure.
    """
    task_registry = load_task_registry()
    task_results = {name: {"status": "pending", "finetune_task": spec["finetune_task"]}
                     for name, spec in task_registry.items()}

    if not accelerator.is_main_process:
        return task_results

    require_finetune_section(pretrain_cfg)
    require_single_process(accelerator)

    # imported lazily to avoid import overhead / cycles for callers that never
    # actually trigger the finetuning chain (e.g. plain inference scripts)
    from scripts.finetune import main as run_single_finetune

    pretrain_output_dir = pretrain_cfg.paths.output_dir
    summary_root = finetune_output_root if finetune_output_root is not None else pretrain_output_dir
    ft_cfg = pretrain_cfg.finetune
    # The seed that will actually drive each fold's torch training run: whatever
    # build_finetune_config resolves cfg.training.seed to (pretrain_cfg.finetune.training.seed
    # if set, else configs/finetune/default.yaml's own default) -- also reused as the outer
    # fold-split random_state, so a task's fold membership and its models' initialization
    # come from the same one seed, mirroring kd_tasks' single seed field.
    seed = ft_cfg.get("training", {}).get(
        "seed", OmegaConf.load(FINETUNE_TEMPLATE_PATH).training.seed)

    raw_cv = OmegaConf.to_container(ft_cfg.get("cv", {}), resolve=True) if ft_cfg.get("cv") else {}
    cv_conf = {**resolve_cv_config(raw_cv), "keep_fold_data": raw_cv.get("keep_fold_data", False)}

    # Loaded once, lazily, the first time a single-study task actually needs pooled data --
    # every single-study task in the registry shares these two AnnData objects instead of
    # each re-reading the full base h5ads.
    pooled_source = {}

    def _pooled_source():
        if not pooled_source:
            logger.info(f"AUTO-FINETUNE: loading base h5ads for pooled CV: "
                       f"{ft_cfg.paths.downstream_train}, {ft_cfg.paths.downstream_test}")
            pooled_source["train"] = ad.read_h5ad(ft_cfg.paths.downstream_train)
            pooled_source["test"] = ad.read_h5ad(ft_cfg.paths.downstream_test)
        return pooled_source["train"], pooled_source["test"]

    # write the all-pending summary immediately, so the file exists from the very start
    write_finetune_summary(summary_root, task_registry, task_results)

    for task_name, task_spec in task_registry.items():
        logger.info("=" * 80)
        logger.info(f"AUTO-FINETUNE: starting task '{task_name}' ({task_spec['finetune_task']})")
        logger.info("=" * 80)
        try:
            skip_reason = None
            if tasks is not None and task_name not in tasks:
                skip_reason = "not in tasks filter"
            else:
                mode, studies = resolve_task_mode(
                    ft_cfg.paths.downstream_train, ft_cfg.paths.downstream_test, task_name,
                    task_spec["label_column"], task_spec.get("ignored_labels", []),
                    declared=task_spec.get("split_mode", "auto"))
                logger.info(f"AUTO-FINETUNE: '{task_name}' split mode: {mode} "
                           f"({len(studies)} distinct study/studies: {studies})")
                if mode_filter is not None and mode != mode_filter:
                    skip_reason = f"mode={mode}, filtered to mode_filter={mode_filter}"
                elif only_fold is not None and mode != "combined_cv":
                    skip_reason = f"mode={mode}, only_fold={only_fold} requires combined_cv (no folds here)"

            if skip_reason is not None:
                logger.info(f"AUTO-FINETUNE: '{task_name}' skipped ({skip_reason})")
                task_results[task_name] = {
                    "status": "skipped",
                    "finetune_task": task_spec["finetune_task"],
                    "reason": skip_reason,
                }
            elif mode == "combined_cv":
                adata_train_full, adata_test_full = _pooled_source()
                result = run_task_cv(
                    pretrain_cfg, pretrain_output_dir, task_name, task_spec,
                    finetune_output_root, cv_conf, seed,
                    adata_train_full, adata_test_full,
                    build_finetune_config, run_single_finetune,
                    only_fold=only_fold,
                )
                task_results[task_name] = {**result, "finetune_task": task_spec["finetune_task"]}
            else:
                task_cfg = build_finetune_config(
                    pretrain_cfg, pretrain_output_dir, task_name, task_spec,
                    finetune_output_root=finetune_output_root,
                )
                run_single_finetune(task_cfg)
                task_results[task_name] = {
                    "status": "ok",
                    "finetune_task": task_spec["finetune_task"],
                    "output_dir": task_cfg.paths.output_dir,
                }
        except Exception as e:
            logger.error(f"AUTO-FINETUNE: task '{task_name}' failed: {e}")
            logger.error(traceback.format_exc())
            task_results[task_name] = {
                "status": "failed",
                "finetune_task": task_spec["finetune_task"],
                "error": str(e),
            }

        # re-written after every task (not just at the end) so a hard kill (walltime,
        # OOM, node failure) partway through still leaves an accurate, current summary
        write_finetune_summary(summary_root, task_registry, task_results)

    num_failed = sum(1 for r in task_results.values() if r["status"] not in ("ok", "skipped"))
    if num_failed:
        logger.error(f"AUTO-FINETUNE: {num_failed}/{len(task_results)} tasks did not complete successfully.")

    return task_results


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _read_task_row(task_dir: Path, task_name: str, finetune_task: str) -> Dict:
    """
    One task's result row, mode-aware: task_dir = .../finetune/<task_name>.

    Checks task_dir/cv_summary.yaml first (pooled K-fold CV for single-study tasks -- see
    utils/finetune_cv_utils.py::run_task_cv), and reports the fold mean in place of a single
    point estimate. Falls back to task_dir/best_model/test_metrics.yaml (presplit,
    multi-study tasks, unchanged).
    """
    cv_path = task_dir / "cv_summary.yaml"
    if cv_path.exists():
        payload = OmegaConf.to_container(OmegaConf.load(cv_path), resolve=True)
        aggregates = payload["aggregates"]
        status = f"OK ({payload['outer_folds']}-fold CV)"

        def mean(*keys):
            for key in keys:
                if key in aggregates:
                    return aggregates[key]["mean"]
            return None

        if finetune_task == "classification":
            return {
                "task": task_name, "type": "classification", "status": status,
                "accuracy": mean("test_accuracy"),
                "f1_macro": mean("test_f1_macro", "test_f1"),
                "f1_weighted": mean("test_f1_weighted"),
                "auroc": mean("test_auroc_weighted", "test_auroc"),
                "mae": None, "rmse": None, "r2": None,
            }
        return {
            "task": task_name, "type": "regression", "status": status,
            "accuracy": None, "f1_macro": None, "f1_weighted": None, "auroc": None,
            "mae": mean("test_mae"), "rmse": mean("test_rmse"), "r2": mean("test_r2"),
        }

    metrics_path = task_dir / "best_model" / "test_metrics.yaml"
    if not metrics_path.exists():
        return {
            "task": task_name, "type": finetune_task, "status": "MISSING",
            "accuracy": None, "f1_macro": None, "f1_weighted": None,
            "auroc": None, "mae": None, "rmse": None, "r2": None,
        }

    metrics = OmegaConf.to_container(OmegaConf.load(metrics_path), resolve=True)
    if finetune_task == "classification":
        return {
            "task": task_name, "type": "classification", "status": "OK",
            "accuracy": metrics.get("test_accuracy"),
            "f1_macro": metrics.get("test_f1_macro", metrics.get("test_f1")),
            "f1_weighted": metrics.get("test_f1_weighted"),
            "auroc": metrics.get("test_auroc_weighted", metrics.get("test_auroc")),
            "mae": None, "rmse": None, "r2": None,
        }
    return {
        "task": task_name, "type": "regression", "status": "OK",
        "accuracy": None, "f1_macro": None, "f1_weighted": None, "auroc": None,
        "mae": metrics.get("test_mae"), "rmse": metrics.get("test_rmse"), "r2": metrics.get("test_r2"),
    }


def write_finetune_summary(pretrain_output_dir: str, task_registry: Dict[str, Dict], task_results: Dict[str, Dict]) -> None:
    """
    Collect every task's result (test_metrics.yaml for presplit tasks, cv_summary.yaml for
    pooled single-study tasks -- see _read_task_row) into one Markdown table at
    <pretrain_output_dir>/finetune_summary.md.

    :param pretrain_output_dir: Output directory of the completed pretraining run.
    :param task_registry: {task_name: {finetune_task, label_column, ignored_labels}}.
    :param task_results: {task_name: {status, finetune_task, output_dir?, error?}}, as
        produced by run_all_downstream_finetunes.
    """
    rows = []
    for task_name in task_registry:
        result = task_results.get(task_name, {"status": "pending"})

        if result["status"] == "pending":
            rows.append({
                "task": task_name,
                "type": result.get("finetune_task", "–"),
                "status": "pending",
                "accuracy": None, "f1_macro": None, "f1_weighted": None,
                "auroc": None, "mae": None, "rmse": None, "r2": None,
            })
            continue

        if result["status"] == "failed":
            rows.append({
                "task": task_name,
                "type": result.get("finetune_task", "–"),
                "status": f"FAILED: {result.get('error', 'unknown error')}",
                "accuracy": None, "f1_macro": None, "f1_weighted": None,
                "auroc": None, "mae": None, "rmse": None, "r2": None,
            })
            continue

        if result["status"] == "skipped":
            rows.append({
                "task": task_name,
                "type": result.get("finetune_task", "–"),
                "status": f"SKIPPED ({result.get('reason', 'mode filter')})",
                "accuracy": None, "f1_macro": None, "f1_weighted": None,
                "auroc": None, "mae": None, "rmse": None, "r2": None,
            })
            continue

        rows.append(_read_task_row(Path(result["output_dir"]), task_name, result["finetune_task"]))

    lines = [
        "# Finetuning summary",
        "",
        f"Pretrained run: `{pretrain_output_dir}`",
        "",
        "| Task | Type | Status | Accuracy | F1 (macro) | F1 (weighted) | AUROC* | MAE | RMSE | R² |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['task']} | {row['type']} | {row['status']} | "
            f"{_fmt(row['accuracy'])} | {_fmt(row['f1_macro'])} | {_fmt(row['f1_weighted'])} | "
            f"{_fmt(row['auroc'])} | {_fmt(row['mae'])} | {_fmt(row['rmse'])} | {_fmt(row['r2'])} |"
        )
    lines.append("")
    lines.append(FINETUNE_AUROC_FOOTNOTE)

    summary_path = Path(pretrain_output_dir) / "finetune_summary.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n")
    logger.info(f"Saved finetuning summary to {summary_path}")

    # Every pretrain config's output_dir is a sibling under the same stage directory
    # (e.g. outputs/pretrain/real_runs/stage2/<config_name>/), so re-scanning that
    # directory after each write keeps a single combined table current across all
    # configs in that stage, including ones finetuned by a different process/job
    # than this one -- and never mixes rows across stages.
    combine_finetune_summaries(str(Path(pretrain_output_dir).parent))


def combine_finetune_summaries(parent_dir: str, out_name: str = None) -> None:
    """
    Aggregate every sibling run's finetune_summary.md directly under parent_dir (e.g.
    outputs/pretrain/real_runs/stage2/) into one combined Markdown table at
    <parent_dir>/<out_name>, one row per (run, task).

    Re-reads each task's test_metrics.yaml directly rather than re-parsing the
    per-run markdown files, so this has no dependency on in-memory task_results from
    other runs' (possibly already-finished) processes -- it only needs what's on disk.

    :param parent_dir: Stage directory containing one subdirectory per pretrain run
        (each subdirectory optionally holding its own finetune_summary.md), e.g.
        outputs/pretrain/real_runs/stage2/.
    :param out_name: Filename for the combined summary, written under parent_dir.
        Defaults to "<parent_dir's own directory name>_finetune_summary_combined.md"
        (e.g. "stage2_finetune_summary_combined.md"), so each stage gets its own
        combined summary automatically.
    """
    parent = Path(parent_dir)
    if out_name is None:
        out_name = f"{parent.name}_finetune_summary_combined.md"
    task_registry = load_task_registry()
    run_dirs = sorted(
        d for d in parent.iterdir()
        if d.is_dir() and (d / "finetune_summary.md").exists()
    )

    rows = []
    for run_dir in run_dirs:
        for task_name, task_spec in task_registry.items():
            task_dir = run_dir / "finetune" / task_name
            row = _read_task_row(task_dir, task_name, task_spec["finetune_task"])
            rows.append({"run": run_dir.name, **row})

    lines = [
        "# Combined finetuning summary",
        "",
        f"Runs: {', '.join(d.name for d in run_dirs) if run_dirs else '(none found)'}",
        "",
        "| Run | Task | Type | Status | Accuracy | F1 (macro) | F1 (weighted) | AUROC* | MAE | RMSE | R² |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['run']} | {row['task']} | {row['type']} | {row['status']} | "
            f"{_fmt(row['accuracy'])} | {_fmt(row['f1_macro'])} | {_fmt(row['f1_weighted'])} | "
            f"{_fmt(row['auroc'])} | {_fmt(row['mae'])} | {_fmt(row['rmse'])} | {_fmt(row['r2'])} |"
        )
    lines.append("")
    lines.append(FINETUNE_AUROC_FOOTNOTE)

    combined_path = parent / out_name
    combined_path.write_text("\n".join(lines) + "\n")
    # Use the underlying stdlib logger (not the accelerate-wrapped one): this function
    # is also meant to be run standalone (e.g. to re-combine existing summaries) where
    # no Accelerator/PartialState has been initialized, which accelerate's logger requires.
    logger.logger.info(f"Saved combined finetuning summary to {combined_path}")


def combine_seeded_finetune_summaries(seeds_root_dir, output_path: Optional[Path] = None) -> None:
    """
    Aggregate one config's presplit (multi-study) finetuning results across repeated
    training seeds into mean +/- (sample) standard deviation tables -- the finetuning
    counterpart to utils/downstream_summary_utils.py::combine_seeded_downstream_summaries,
    adapted for finetuning's directory layout (one neural net per task, not one per
    classical-ML method).

    Expects the layout produced by repeating run_all_downstream_finetunes with
    mode_filter="presplit" once per seed (see scripts/run_finetune_pretrained_with_seeds.sh /
    run_finetune_scratch_with_seeds.sh):
        <seeds_root_dir>/seed_<i>/finetune/<task>/...

    Tasks with no result in any seed (e.g. single-study tasks correctly excluded by
    mode_filter="presplit") are skipped entirely rather than rendered as all-missing rows.

    :param seeds_root_dir: Directory containing one subdirectory per seed (named "seed_<i>"),
        e.g. outputs/pretrain/finetune_pretrained_seeds/<config_name>_multi_study_seeds/.
    :param output_path: Where to write the resulting summary; defaults to
        <seeds_root_dir>/combined_summary_with_error_bars.md.
    """
    seeds_root_dir = Path(seeds_root_dir)
    output_path = Path(output_path) if output_path else seeds_root_dir / "combined_summary_with_error_bars.md"

    seed_dirs = sorted(d for d in seeds_root_dir.iterdir() if d.is_dir() and d.name.startswith("seed_"))
    if not seed_dirs:
        logger.logger.warning(f"No seed_* directories found under {seeds_root_dir}")
        return

    task_registry = load_task_registry()

    lines = [
        "# Combined finetuning summary (mean ± std across seeds)",
        "",
        f"Seeds: {', '.join(d.name for d in seed_dirs)} (n={len(seed_dirs)})",
        "",
        "Each cell is the sample mean ± sample standard deviation of the metric across the "
        "seeds above -- each seed re-finetunes from the same checkpoint with a different "
        "training seed (see utils/finetune_orchestration.py::run_all_downstream_finetunes' "
        "mode_filter). A cell with no `±` means only one seed produced a result for that "
        "task; tasks with no result in any seed (e.g. single-study tasks, correctly excluded "
        "from this seeded sweep) are omitted below.",
        "",
    ]

    rows_by_metric = {metric_key: [] for metric_key in METRIC_TITLES}
    for task_name, task_spec in task_registry.items():
        per_seed_rows = [_read_task_row(seed_dir / "finetune" / task_name, task_name,
                                        task_spec["finetune_task"])
                         for seed_dir in seed_dirs]
        if all(row["status"] == "MISSING" for row in per_seed_rows):
            continue
        for metric_key in METRIC_TITLES:
            mean_std = _mean_std([row[metric_key] for row in per_seed_rows])
            rows_by_metric[metric_key].append({"task": task_name, "values": {"finetune": mean_std}})

    if not any(rows_by_metric.values()):
        lines.append("(no presplit/multi-study finetuning results found on disk)")
        output_path.write_text("\n".join(lines) + "\n")
        return

    lines += _render_error_bar_tables(rows_by_metric, ["finetune"])
    lines.append(FINETUNE_AUROC_FOOTNOTE)

    output_path.write_text("\n".join(lines) + "\n")
    logger.logger.info(f"Saved seeded combined finetuning summary to {output_path}")
