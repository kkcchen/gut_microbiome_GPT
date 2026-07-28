"""
Aggregates downstream-task eval results across repeated train/test-split seeds into
mean +/- std tables (i.e. error bars), since the pipeline itself only ever evaluates
on one fixed split at a time.

Expects <seeds-root>/seed_<i>/<config_name>/downstream_tasks/... for each seed produced
by scripts/run_raw_baselines_with_seeds.sh (or an equivalent manual sweep).
"""
import argparse
from pathlib import Path

from utils.downstream_summary_utils import combine_seeded_downstream_summaries


def main():
    parser = argparse.ArgumentParser(description="Aggregate downstream eval results across seeded splits into mean +/- std tables.")
    parser.add_argument("--seeds-root", type=str, required=True, help="Directory containing one subdirectory per seed (named 'seed_<i>').")
    parser.add_argument("--output-path", type=str, default=None, help="Where to write the summary; defaults to <seeds-root>/combined_summary_with_error_bars.md.")
    args = parser.parse_args()

    combine_seeded_downstream_summaries(
        Path(args.seeds_root),
        Path(args.output_path) if args.output_path else None,
    )


if __name__ == "__main__":
    main()
