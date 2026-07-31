"""
Turns the single, pre-aggregated downstream_agp.h5ad into the (train, test) h5ad pair
every downstream pipeline (utils/downstream_utils.py, utils/finetune_orchestration.py)
expects as its input convention.

Every task in downstream_agp.h5ad is single-study (obs['study_id'] has exactly one
distinct value), so utils/downstream_split_mode.py::resolve_split_mode will resolve all
of them to "combined_cv" -- train and test get pooled back together immediately
(materialize_pooled_task_adata / run_task_combined_cv / run_task_cv) before being
re-split into outer CV folds. Any train/test partition of the AGP file is therefore
numerically equivalent post-pooling, so the exact split ratio doesn't matter -- but it must
be a REAL split: an empty test file crashes embedding extraction's inference-statistics
logging (utils/data_pipeline.py::print_inference_statistics calls .min()/.max() on the
per-sample sequencing depth array with no empty-array guard, since normal inference runs
never hand it zero rows). A plain 80/20 row-level split avoids that, and comfortably
keeps every task's ~4,200-4,541 rows non-empty on both sides.

Usage:
    python scripts/split_agp_downstream_tasks.py
    python scripts/split_agp_downstream_tasks.py --agp-path /path/to/downstream_agp.h5ad
"""
import argparse
import os

import anndata as ad
from sklearn.model_selection import train_test_split

DEFAULT_AGP_PATH = "/scratch/kchen13/hmc_final_fixed/downstream_agp.h5ad"
TEST_SIZE = 0.2
SEED = 42


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agp-path", default=DEFAULT_AGP_PATH,
                        help="Path to the single pre-aggregated AGP h5ad.")
    args = parser.parse_args()

    base, ext = os.path.splitext(args.agp_path)
    train_path, test_path = f"{base}_train{ext}", f"{base}_test{ext}"

    print(f"Reading {args.agp_path}")
    adata = ad.read_h5ad(args.agp_path)
    print(f"  {adata.n_obs} obs x {adata.n_vars} var")
    print(f"  downstream_task values: {sorted(adata.obs['downstream_task'].unique())}")

    train_idx, test_idx = train_test_split(
        range(adata.n_obs), test_size=TEST_SIZE, random_state=SEED)

    train_adata = adata[train_idx].copy()
    test_adata = adata[test_idx].copy()

    train_adata.write_h5ad(train_path)
    print(f"Wrote {train_adata.n_obs} obs to {train_path}")

    test_adata.write_h5ad(test_path)
    print(f"Wrote {test_adata.n_obs} obs to {test_path}")


if __name__ == "__main__":
    main()
