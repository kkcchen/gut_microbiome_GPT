"""
Generator for the "raw baselines" eval configs under
configs/eval/raw_baselines/.

For each of 4 normalization strategies applied directly to the raw
(un-embedded) downstream train/test h5ads, generates a config that runs
the SAME XGBoost grid search used for the embedding baselines
(utils/downstream_models_utils.py::train_xgboost, search_type="grid")
plus a small ElasticNet grid (train_linear, search_type="grid") on the
raw features -- no model forward pass (eval.with_model=False).

This is a floor/control baseline: how well do classical ML methods do on
raw taxa abundances (with different normalizations) with no learned
representation at all, compared to the embedding-baseline sweep in
../embedding_baselines/.

Normalizations (utils/downstream_data_utils.py::normalize_embeddings):
  - rel_ab:            closure(x + 1e-8)                      [compositional, unlogged]
  - log_rel_abundance: log(closure(x + 1e-8))                 [matches pretrain-side
                         data_utils/collator.py::apply_normalization's winning strategy]
  - clr:               clr(closure(x + 1e-8))                 [existing default in raw.yaml]
  - none:               x, unnormalized                        [floor control]

Rerun after editing this file to regenerate every config + config_list.txt:

    python configs/eval/raw_baselines/generate_configs.py
"""
import copy
import os
import sys

from omegaconf import OmegaConf

SWEEP_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDING_BASELINES_DIR = os.path.join(os.path.dirname(SWEEP_DIR), "embedding_baselines")
sys.path.insert(0, EMBEDDING_BASELINES_DIR)
from generate_configs import DOWNSTREAM_TASKS  # noqa: E402  (reuse the same 11 classification tasks)

NORMALIZATIONS = ["rel_ab", "log_rel_abundance", "clr", "none"]

BASE = {
    "paths": {
        "output_dir": None,  # filled per-config
        "train_path": "/scratch/kchen13/hmc_final_fixed/downstream_train.h5ad",
        "test_path": "/scratch/kchen13/hmc_final_fixed/downstream_test.h5ad",
    },
    "eval": {
        "seed": 42,
        "with_model": False,
    },
    "downstream_tasks_config": {
        "normalization": None,  # filled per-config
        "prevalence_threshold": 0.01,
        "abundance_threshold": 0.05,
        "methods": {
            "xgboost": {"search_type": "grid"},
            "linear": {"search_type": "grid"},
        },
        "tasks": DOWNSTREAM_TASKS,
    },
}


def make_config(normalization):
    cfg = copy.deepcopy(BASE)
    cfg["paths"]["output_dir"] = f"outputs/eval/raw_baselines/{normalization}"
    cfg["downstream_tasks_config"]["normalization"] = normalization
    return cfg


def write_config(cfg, filename):
    path = os.path.join(SWEEP_DIR, filename)
    OmegaConf.save(OmegaConf.create(cfg), path)
    return path


def write_list(paths, filename):
    path = os.path.join(SWEEP_DIR, filename)
    with open(path, "w") as f:
        for p in paths:
            f.write(f"configs/eval/raw_baselines/{p}\n")
    return path


def generate():
    files = []
    for normalization in NORMALIZATIONS:
        cfg = make_config(normalization)
        filename = f"{normalization}.yaml"
        write_config(cfg, filename)
        files.append(filename)
    write_list(files, "config_list.txt")
    print(f"Raw baselines: wrote {len(files)} configs")


if __name__ == "__main__":
    generate()
