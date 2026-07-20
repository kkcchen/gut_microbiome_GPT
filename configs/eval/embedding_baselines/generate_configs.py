"""
Generator for the "embedding baselines" eval configs under
configs/eval/embedding_baselines/.

For each of the two winning pretrain checkpoints, generates a config that,
on a single `python -m scripts.eval --config <path>` run, extracts that
checkpoint's output embeddings for the downstream train/test h5ads and then
trains/evaluates classical ML baselines (xgboost, random_forest, linear,
tabpfn) directly on those embeddings, per configs/eval/downstream_probe_template.yaml.

Targets:
  - stage3_abundance_emb_concatenation: overall best architecture from the
    Stage 1-3 sweep (../pretrain/real_runs/stage3_abundance_emb_concatenation.yaml)
  - pretrained_baseline_bins20_mask030: best default-scGPT-style architecture
    (../pretrain/pretrained_baseline/pretrained_baseline_bins20_mask030.yaml)

Each target's `data`/`model.params` fields below are copied from that
checkpoint's own pretrain config -- they must match what the model was
trained with.

Rerun after editing this file to regenerate every config + config_list.txt:

    python configs/eval/embedding_baselines/generate_configs.py
"""
import copy
import os

from omegaconf import OmegaConf

SWEEP_DIR = os.path.dirname(os.path.abspath(__file__))

DOWNSTREAM_TASKS = {
    "age": {"label_type": "continuous_label", "ignored_labels": [], "target_col": "downstream_task"},
    "bmi": {"label_type": "continuous_label", "ignored_labels": [], "target_col": "downstream_task"},
    "sex": {"label_type": "categorical_label", "ignored_labels": [], "target_col": "downstream_task"},
    "supplement": {"label_type": "categorical_label", "ignored_labels": [], "target_col": "downstream_task"},
    "helicobacter_pylori_infection": {"label_type": "categorical_label", "ignored_labels": [], "target_col": "downstream_task"},
    "hbv_infection": {"label_type": "categorical_label", "ignored_labels": [], "target_col": "downstream_task"},
    "hiv_infection": {"label_type": "categorical_label", "ignored_labels": [], "target_col": "downstream_task"},
    "sunflower_seed_oil_emollient_therapy": {"label_type": "categorical_label", "ignored_labels": ["extraction_control"], "target_col": "downstream_task"},
    "chemotherapy": {"label_type": "categorical_label", "ignored_labels": [], "target_col": "downstream_task"},
    "type_1_diabetes": {"label_type": "categorical_label", "ignored_labels": [], "target_col": "downstream_task"},
    "crohns_disease": {"label_type": "categorical_label", "ignored_labels": [], "target_col": "downstream_task"},
    "diarrhea": {"label_type": "categorical_label", "ignored_labels": [], "target_col": "downstream_task"},
    "location": {"label_type": "categorical_label", "ignored_labels": [], "target_col": "downstream_task"},
}

BASE = {
    "paths": {
        "pretrained_model_dir": None,  # filled per-config
        "checkpoint_path": None,  # filled per-config
        "output_dir": None,  # filled per-config
        "eval_files": [
            "/scratch/kchen13/hmc_final_fixed/downstream_train.h5ad",
            "/scratch/kchen13/hmc_final_fixed/downstream_test.h5ad",
        ],
    },
    "eval": {
        "seed": 42,
        "overwrite_output_dir": True,
        "embedding_type": "cls",
        "return_full_output": False,
        "save_embeddings": True,
        "save_metadata": True,
        "tasks": ["masking"],  # unused for embedding extraction, kept for build_model_config compatibility
        "eval_on_downstream_tasks": True,
        "with_model": True,
    },
    "data": {
        "batch_size": 64,
        "norm_strategy": None,  # filled per-config
        "num_bins": None,  # filled per-config
        "split_key": "study_id",
        "use_batch_labels": None,  # filled per-config
        "num_workers": 4,
    },
    "model": {
        "params": {
            "d_model": 128,
            "nhead": 8,
            "d_hid": 512,
            "nlayers": 3,
            "dropout": 0.1,
            "abundance_emb_style": None,  # filled per-config
            "use_gnn": False,
            "sample_emb_style": None,  # filled per-config
        }
    },
    "downstream_tasks_config": {
        "methods": {
            "xgboost": {"search_type": "grid"},
            "random_forest": {"search_type": "grid"},
            "linear": {"search_type": "grid"},
        },
        "tasks": DOWNSTREAM_TASKS,
    },
}

# Each target's architecture/data fields, copied from its own pretrain config.
TARGETS = {
    "stage3_abundance_emb_concatenation": {
        "checkpoint_dir": "outputs/pretrain/real_runs/stage3/stage3_abundance_emb_concatenation",
        "data": {
            "norm_strategy": "log_rel_abundance",
            "num_bins": 10,
            "use_batch_labels": True,
        },
        "model_params": {
            "abundance_emb_style": "concatenation",
            "sample_emb_style": "cls",
        },
    },
    "pretrained_baseline_bins20_mask030": {
        "checkpoint_dir": "outputs/pretrain/pretrained_baseline/pretrained_baseline_bins20_mask030",
        "data": {
            "norm_strategy": "binning",
            "num_bins": 20,
            "use_batch_labels": True,
            "max_seq_len": 200,
        },
        "model_params": {
            "abundance_emb_style": "continuous",
            "sample_emb_style": "cls",
        },
    },
}


def make_config(name, target):
    cfg = copy.deepcopy(BASE)
    checkpoint_dir = target["checkpoint_dir"]
    cfg["paths"]["pretrained_model_dir"] = checkpoint_dir
    cfg["paths"]["checkpoint_path"] = f"{checkpoint_dir}/best_model/best_model/pytorch_model.bin"
    cfg["paths"]["output_dir"] = f"outputs/eval/embedding_baselines/{name}"
    cfg["data"].update(target["data"])
    cfg["model"]["params"].update(target["model_params"])
    return cfg


def write_config(cfg, filename):
    path = os.path.join(SWEEP_DIR, filename)
    OmegaConf.save(OmegaConf.create(cfg), path)
    return path


def write_list(paths, filename):
    path = os.path.join(SWEEP_DIR, filename)
    with open(path, "w") as f:
        for p in paths:
            f.write(f"configs/eval/embedding_baselines/{p}\n")
    return path


def generate():
    files = []
    for name, target in TARGETS.items():
        cfg = make_config(name, target)
        filename = f"{name}.yaml"
        write_config(cfg, filename)
        files.append(filename)
    write_list(files, "config_list.txt")
    print(f"Embedding baselines: wrote {len(files)} configs")


if __name__ == "__main__":
    generate()
