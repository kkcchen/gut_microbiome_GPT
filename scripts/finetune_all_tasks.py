"""
Re-runs the finetune-all-tasks chain (utils/finetune_orchestration.py) against an
ALREADY-TRAINED pretraining checkpoint, without touching or repeating pretraining.

scripts/train.py only triggers finetune_orchestration.run_all_downstream_finetunes()
as the tail end of a pretraining run, reading/writing everything under that same
run's paths.output_dir. This script calls the same function directly against an
existing checkpoint, so you can re-finetune it against a different downstream
train/test split (e.g. a different scripts/preprocess.py seed) and redirect the
results elsewhere with --finetune-output-root, instead of overwriting that
checkpoint's original finetune_summary.md/finetune/<task>/ results.

Used by scripts/run_finetune_pretrained_with_seeds.sh to compute error bars across
repeated split seeds for the "finetuned from pretrained checkpoint" results.
"""
import argparse

from accelerate import Accelerator

from utils.config_utils import load_and_validate_config
from utils.finetune_orchestration import (
    require_finetune_section,
    require_single_process,
    run_all_downstream_finetunes,
)
from trainers import logger


def main():
    parser = argparse.ArgumentParser(
        description="Re-finetune an existing pretrained checkpoint on all registered downstream tasks."
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to the pretrain YAML config that produced the checkpoint (same one used with scripts.train).",
    )
    parser.add_argument(
        "--finetune-output-root", type=str, default=None,
        help="Directory to write results under instead of the pretrain config's own paths.output_dir "
             "(the checkpoint/vocab are still read from paths.output_dir either way).",
    )
    parser.add_argument(
        "--mode-filter", type=str, default=None, choices=["combined_cv", "presplit"],
        help="Only run tasks whose auto-detected split mode matches (see "
             "utils/downstream_split_mode.py::resolve_split_mode); others are recorded as "
             "'skipped'. Used to repeat only presplit (multi-study) tasks across several "
             "training seeds without redoing single-study tasks' already-real pooled-CV "
             "error bars -- see scripts/run_finetune_pretrained_with_seeds.sh.",
    )
    parser.add_argument(
        "--fold", type=int, default=None,
        help="Only train this one outer fold (0-indexed) of every combined_cv task, caching "
             "it, and skip presplit tasks entirely (see utils/finetune_orchestration.py::"
             "run_all_downstream_finetunes' only_fold param). Run again without --fold once "
             "every fold of every task is done, to assemble cv_summary.yaml -- that pass "
             "cache-hits every fold, so it's fast. Lets a task's outer folds run as separate "
             "parallel jobs, which matters far more here than in the classical-ML pipelines "
             "since each fold is a full finetuning run.",
    )
    parser.add_argument(
        "--tasks", type=str, default=None,
        help="Comma-separated task names to run (must match keys in "
             "configs/finetune/task_registry.yaml); others are recorded as 'skipped'. Needed "
             "when the registry holds tasks from more than one dataset -- "
             "finetune.paths.downstream_train/downstream_test are global to the whole run, so "
             "this keeps a run scoped to the tasks that actually live in the data those paths "
             "point at. A single string (not nargs='*') because this parser also has a "
             "REMAINDER positional below for config overrides -- nargs='*' would greedily "
             "swallow those too.",
    )
    parser.add_argument(
        "overrides", nargs=argparse.REMAINDER,
        help="Override config values (e.g. finetune.paths.downstream_train=/path/to/seed/downstream_train.h5ad)",
    )
    args = parser.parse_args()
    tasks = [t.strip() for t in args.tasks.split(",")] if args.tasks else None

    cfg = load_and_validate_config(args.config, args.overrides)
    require_finetune_section(cfg)

    accelerator = Accelerator()
    require_single_process(accelerator)

    logger.info("=" * 80)
    logger.info(f"RE-FINETUNING EXISTING CHECKPOINT: {cfg.paths.output_dir}")
    if args.finetune_output_root:
        logger.info(f"Results will be written under: {args.finetune_output_root}")
    logger.info("=" * 80)

    task_results = run_all_downstream_finetunes(cfg, accelerator, finetune_output_root=args.finetune_output_root,
                                                mode_filter=args.mode_filter, tasks=tasks,
                                                only_fold=args.fold)
    accelerator.wait_for_everyone()
    accelerator.end_training()

    if accelerator.is_main_process:
        num_failed = sum(1 for r in task_results.values() if r["status"] not in ("ok", "skipped"))
        if num_failed:
            logger.error(
                f"{num_failed}/{len(task_results)} finetune tasks did not complete successfully -- "
                "see finetune_summary.md and the logs above for details."
            )
            raise SystemExit(1)


if __name__ == "__main__":
    main()
