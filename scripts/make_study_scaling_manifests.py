"""
Generate the v2 data-scaling split manifests: one fixed study-isolated validation
holdout (whole BioProjects, ~10% of samples) shared by every ratio, plus nested
stratified-by-study training subsets at 10 / 20 / 50 / 100 % of the train pool.

    source ~/hmbenv/bin/activate
    python scripts/make_study_scaling_manifests.py

Writes scaling_splits/study_manifest_r0{10,20,50,100}.json in the schema
utils/data_pipeline.load_scaling_split expects:
    {"n_total": N, "splits": {"<key>": {"train": [int...], "val": [int...]}}, ...}
`val` is byte-for-byte identical across the four files. `train` sets are nested:
train_r010 ⊂ train_r020 ⊂ train_r050 ⊂ train_r100, and train_r100 = the whole pool.
Every training study appears at every ratio (≥1 sample) so the pretraining
BatchVocabulary — hence the model's batch-embedding table — is identical for all runs.

Reproducible: np.random.default_rng(SEED).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.downstream_split_mode import read_obs  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANN_PATH = "/scratch/kchen13/hmc_final_fixed/pretrain.h5ad"
OUT_DIR = os.path.join(ROOT, "scaling_splits")
SEED = 42
VAL_FRAC = 0.10
RATIOS = [0.10, 0.20, 0.50, 1.00]
KEY = {0.10: "r010", 0.20: "r020", 0.50: "r050", 1.00: "r100"}


def pick_val_studies(study_ids, rng, val_frac):
    """Greedy whole-study accumulation to ~val_frac of samples (mirrors
    utils/data_pipeline.split_data's grouped branch, but with a seeded rng)."""
    studies, counts = np.unique(study_ids, return_counts=True)
    count_of = dict(zip(studies, counts))
    order = rng.permutation(studies)
    target = int(len(study_ids) * val_frac)
    val, n = [], 0
    for s in order:
        if n < target:
            val.append(s); n += count_of[s]
        elif abs(n + count_of[s] - target) < abs(n - target):
            val.append(s); n += count_of[s]
    return set(val)


def main():
    obs = read_obs(ANN_PATH)
    study = obs["study_id"].to_numpy().astype(str)
    n_total = len(study)
    print(f"{ANN_PATH}: {n_total} samples, {len(np.unique(study))} studies")

    rng = np.random.default_rng(SEED)

    val_studies = pick_val_studies(study, rng, VAL_FRAC)
    is_val = np.isin(study, list(val_studies))
    val_idx = np.nonzero(is_val)[0]
    pool_idx = np.nonzero(~is_val)[0]
    pool_studies = sorted(set(study[pool_idx]))
    print(f"val: {len(val_studies)} studies / {len(val_idx)} samples "
          f"({len(val_idx)/n_total:.1%})")
    print(f"train pool: {len(pool_studies)} studies / {len(pool_idx)} samples")

    # Fixed per-study shuffle of the pool; each ratio takes a prefix -> nested subsets.
    per_study_perm = {}
    for s in pool_studies:
        idx = np.nonzero(study == s)[0]
        per_study_perm[s] = idx[rng.permutation(len(idx))]

    train_sets = {}
    for r in RATIOS:
        parts = []
        for s in pool_studies:
            perm = per_study_perm[s]
            k = len(perm) if r >= 1.0 else max(1, int(np.round(r * len(perm))))
            parts.append(perm[:k])
        train_sets[r] = np.sort(np.concatenate(parts))

    # ---- assertions ----
    val_set = set(val_idx.tolist())
    prev = set()
    for r in RATIOS:
        tr = set(train_sets[r].tolist())
        assert tr.isdisjoint(val_set), f"r={r}: train ∩ val non-empty"
        assert prev <= tr, f"r={r}: not nested over previous ratio"
        assert set(study[train_sets[r]]) == set(pool_studies), \
            f"r={r}: not all pool studies represented"
        prev = tr
    assert train_sets[1.0].tolist() == sorted(pool_idx.tolist()), "r100 != whole pool"
    assert not (set(study[val_idx]) & set(pool_studies)), "val/train studies overlap"

    # ---- write ----
    os.makedirs(OUT_DIR, exist_ok=True)
    val_list = val_idx.tolist()
    for r in RATIOS:
        key = KEY[r]
        manifest = {
            "n_total": n_total,
            "seed": SEED,
            "val_frac": VAL_FRAC,
            "ratio": r,
            "study_isolated_val": True,
            "stratified_by_study": True,
            "n_val_studies": len(val_studies),
            "n_train_studies": len(pool_studies),
            "n_train": int(len(train_sets[r])),
            "n_val": len(val_list),
            "splits": {key: {"train": train_sets[r].tolist(), "val": val_list}},
        }
        path = os.path.join(OUT_DIR, f"study_manifest_{key}.json")
        with open(path, "w") as f:
            json.dump(manifest, f)
        print(f"  wrote {path}  train={manifest['n_train']:>6d}  "
              f"({manifest['n_train']/len(pool_idx):.1%} of pool)")

    print("\nnesting / isolation / study-coverage assertions all passed.")


if __name__ == "__main__":
    main()
