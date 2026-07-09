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
from typing import Dict

from omegaconf import OmegaConf

from trainers import logger

TASK_REGISTRY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs", "finetune", "task_registry.yaml",
)
FINETUNE_TEMPLATE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs", "finetune", "default.yaml",
)


def load_task_registry(path: str = TASK_REGISTRY_PATH) -> Dict[str, Dict]:
    """Load the {task_name: {finetune_task, label_column}} registry."""
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


def build_finetune_config(pretrain_cfg, pretrain_output_dir: str, task_name: str, task_spec: Dict):
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
    :return: OmegaConf config ready to pass to scripts.finetune.main().
    """
    require_finetune_section(pretrain_cfg)
    cfg = OmegaConf.load(FINETUNE_TEMPLATE_PATH)
    ft_cfg = pretrain_cfg.finetune

    task_output_dir = os.path.join(pretrain_output_dir, "finetune", task_name)

    cfg.paths.downstream_train = ft_cfg.paths.downstream_train
    cfg.paths.downstream_test = ft_cfg.paths.get("downstream_test", None)
    cfg.paths.output_dir = task_output_dir
    cfg.paths.model_config_path = os.path.join(task_output_dir, "model_config.json")
    cfg.paths.taxa_vocab_path = os.path.join(pretrain_output_dir, "taxa_vocab.pkl")
    cfg.paths.checkpoint_path = os.path.join(
        pretrain_output_dir, "best_model", "best_model", "pytorch_model.bin"
    )

    # Disabled by default: a wandb run per task per pretrain config would multiply
    # runs 12x; enable manually on the template if per-task tracking is wanted.
    cfg.wandb.enabled = False

    cfg.data.norm_strategy = pretrain_cfg.data.norm_strategy
    cfg.data.use_batch_labels = pretrain_cfg.data.use_batch_labels
    cfg.data.max_seq_len = pretrain_cfg.data.max_seq_len
    cfg.data.num_bins = pretrain_cfg.data.get("num_bins", 15)
    cfg.data.num_workers = pretrain_cfg.data.get("num_workers", 4)
    cfg.data.val_size = ft_cfg.get("val_size", 0.2)
    cfg.data.finetune_task_name = task_name
    cfg.data.label_column = task_spec["label_column"]

    for key, value in ft_cfg.get("training", {}).items():
        cfg.training[key] = value
    cfg.training.finetune_task = task_spec["finetune_task"]

    cfg.model.params = copy.deepcopy(pretrain_cfg.model.params)

    return cfg


def run_all_downstream_finetunes(pretrain_cfg, accelerator) -> None:
    """
    Finetune the just-completed pretraining run on every task in the task
    registry, then write an aggregated summary. No-ops on non-main processes.

    :param pretrain_cfg: OmegaConf config of the pretraining run that just finished.
    :param accelerator: Accelerator used for pretraining (checked for is_main_process).
    """
    if not accelerator.is_main_process:
        return

    require_finetune_section(pretrain_cfg)

    # imported lazily to avoid import overhead / cycles for callers that never
    # actually trigger the finetuning chain (e.g. plain inference scripts)
    from scripts.finetune import main as run_single_finetune

    pretrain_output_dir = pretrain_cfg.paths.output_dir
    task_registry = load_task_registry()

    task_results = {}
    for task_name, task_spec in task_registry.items():
        logger.info("=" * 80)
        logger.info(f"AUTO-FINETUNE: starting task '{task_name}' ({task_spec['finetune_task']})")
        logger.info("=" * 80)
        try:
            task_cfg = build_finetune_config(pretrain_cfg, pretrain_output_dir, task_name, task_spec)
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

    write_finetune_summary(pretrain_output_dir, task_registry, task_results)


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_finetune_summary(pretrain_output_dir: str, task_registry: Dict[str, Dict], task_results: Dict[str, Dict]) -> None:
    """
    Collect every task's test_metrics.yaml (written by
    trainers.trainer.MicrobiomeTrainer.evaluate_on_test_set) into one Markdown
    table at <pretrain_output_dir>/finetune_summary.md.

    :param pretrain_output_dir: Output directory of the completed pretraining run.
    :param task_registry: {task_name: {finetune_task, label_column}}.
    :param task_results: {task_name: {status, finetune_task, output_dir?, error?}}, as
        produced by run_all_downstream_finetunes.
    """
    rows = []
    for task_name in task_registry:
        result = task_results.get(task_name, {"status": "failed", "error": "not run"})

        if result["status"] == "failed":
            rows.append({
                "task": task_name,
                "type": result.get("finetune_task", "–"),
                "status": f"FAILED: {result.get('error', 'unknown error')}",
                "accuracy": None, "f1_macro": None, "f1_weighted": None,
                "auroc": None, "mae": None, "rmse": None, "r2": None,
            })
            continue

        metrics_path = Path(result["output_dir"]) / "best_model" / "test_metrics.yaml"
        if not metrics_path.exists():
            rows.append({
                "task": task_name,
                "type": result["finetune_task"],
                "status": f"FAILED: test_metrics.yaml not found at {metrics_path}",
                "accuracy": None, "f1_macro": None, "f1_weighted": None,
                "auroc": None, "mae": None, "rmse": None, "r2": None,
            })
            continue

        metrics = OmegaConf.to_container(OmegaConf.load(metrics_path), resolve=True)
        if result["finetune_task"] == "classification":
            auroc = metrics.get("test_auroc_weighted", metrics.get("test_auroc"))
            f1_macro = metrics.get("test_f1_macro", metrics.get("test_f1"))
            rows.append({
                "task": task_name,
                "type": "classification",
                "status": "OK",
                "accuracy": metrics.get("test_accuracy"),
                "f1_macro": f1_macro,
                "f1_weighted": metrics.get("test_f1_weighted"),
                "auroc": auroc,
                "mae": None, "rmse": None, "r2": None,
            })
        else:
            rows.append({
                "task": task_name,
                "type": "regression",
                "status": "OK",
                "accuracy": None, "f1_macro": None, "f1_weighted": None, "auroc": None,
                "mae": metrics.get("test_mae"),
                "rmse": metrics.get("test_rmse"),
                "r2": metrics.get("test_r2"),
            })

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
    lines.append(
        "*AUROC is the class-weighted (support-weighted) macro-average across classes "
        'for multi-class tasks (sklearn `average="weighted"`); the standard AUROC for binary tasks.'
    )

    summary_path = Path(pretrain_output_dir) / "finetune_summary.md"
    summary_path.write_text("\n".join(lines) + "\n")
    logger.info(f"Saved finetuning summary to {summary_path}")
