"""
Aggregates presplit (multi-study) finetuning results across repeated training seeds into
mean +/- std tables (i.e. error bars) -- the finetuning counterpart to
scripts/aggregate_seeded_downstream_summary.py.

Expects <seeds-root>/seed_<i>/finetune/<task>/... for each seed, produced by running
run_all_downstream_finetunes with mode_filter="presplit" once per seed (see
scripts/run_finetune_pretrained_with_seeds.sh / run_finetune_scratch_with_seeds.sh).
"""
import argparse
from pathlib import Path

from utils.finetune_orchestration import combine_seeded_finetune_summaries


def main():
    parser = argparse.ArgumentParser(description="Aggregate presplit finetuning results across seeded reruns into mean +/- std tables.")
    parser.add_argument("--seeds-root", type=str, required=True, help="Directory containing one subdirectory per seed (named 'seed_<i>').")
    parser.add_argument("--output-path", type=str, default=None, help="Where to write the summary; defaults to <seeds-root>/combined_summary_with_error_bars.md.")
    args = parser.parse_args()

    combine_seeded_finetune_summaries(
        Path(args.seeds_root),
        Path(args.output_path) if args.output_path else None,
    )


if __name__ == "__main__":
    main()
