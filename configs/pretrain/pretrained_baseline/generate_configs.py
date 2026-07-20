"""
Generator for the "pretrained default-scGPT baseline" sweep under
configs/pretrain/pretrained_baseline/.

Same model/data architecture as
configs/pretrain/real_runs/baseline_random_init_default_scgpt.yaml (the
"default scGPT-style architecture": norm_strategy=binning, sample_emb_style=cls,
abundance_emb_style=continuous, use_batch_labels=true, tasks=[masking,
masking_from_cls]), but unlike that config this one actually pretrains
(training.max_epochs=200, like Stage 1-4 in ../real_runs/) instead of skipping
straight to a random-init encoder, and downstream-finetunes with the sweep's
standard linear-probe protocol (finetune.training.finetune_mode="partial")
instead of the baseline's full-finetune ablation protocol.

Swept over a num_bins x masking_prob grid (3x3=9 configs), mirroring
../real_runs/stage4's grid shape but holding the task family fixed at
[masking, masking_from_cls] instead of Stage 2's winning task.

Rerun after editing this file to regenerate every config + config_list.txt:

    python configs/pretrain/pretrained_baseline/generate_configs.py
"""
import copy
import os

from omegaconf import OmegaConf

SWEEP_DIR = os.path.dirname(os.path.abspath(__file__))

BASE = {
    "paths": {
        "output_dir": None,  # filled per-config
        "model_config_path": "${paths.output_dir}/model_config.json",
        "ann_table_path": "/scratch/kchen13/hmc_final_fixed/pretrain.h5ad",
    },
    "checkpoint_path": None,
    "wandb": {
        "enabled": True,
        "entity": "haoze-deng-university-of-toronto",
        "project": "microbiome-pretrain-sweep-pretrained-baseline",
        "run_name": None,  # filled per-config
        "run_notes": None,  # filled per-config
    },
    "training": {
        "seed": 42,
        "init_lr": 1e-3,
        "batch_size": 64,
        "max_epochs": 200,
        "cosine_warmup_ratio_or_step": 0.1,
        "log_interval": 10,
        "patience": 20,
        "grad_accumulation_steps": 1,
        "enable_fp16": False,
        "notes": None,  # filled per-config
        "tasks": ["masking", "masking_from_cls"],
        "masking_prob": None,  # filled per-config
        "masking_taxa_prob": 0.15,
        "optimizer": "adamw",
        "lr_scheduler": "cosine",
        "grad_clip": 1.0,
        "checkpoint_every": 5,
        "finetune_mode": "none",  # keeps the shared trainer on the pretraining path
    },
    "validation": {
        "batch_size": 64,
        "eval_interval_epochs": 1,
    },
    "data": {
        "norm_strategy": "binning",
        "num_bins": None,  # filled per-config
        "split_key": "study_id",
        "val_size": 0.1,
        "use_batch_labels": True,
        "max_seq_len": 200,
        "num_workers": 4,
        "metadata_fields": [],
    },
    "model": {
        "params": {
            "d_model": 128,
            "nhead": 8,
            "d_hid": 512,
            "nlayers": 3,
            "dropout": 0.1,
            "abundance_emb_style": "continuous",
            "use_gnn": False,
            "sample_emb_style": "cls",
            "model_distribution": None,
            "seq_len": None,
            "preinitialized_taxa_embedding_path": None,
            "freeze_preinitialized_embeddings": False,
            "preinitialized_embedding_projection": "linear",
        }
    },
    "debug": {
        "nrows": None,
        "start_over": True,
    },
    "finetune": {
        "paths": {
            "downstream_train": "/scratch/kchen13/hmc_final_fixed/downstream_train.h5ad",
            "downstream_test": "/scratch/kchen13/hmc_final_fixed/downstream_test.h5ad",
        },
        "training": {
            "init_lr": 1e-4,
            "max_epochs": 10,
            "patience": 10,
            "finetune_mode": "partial",
            "batch_size": 64,
            "use_class_weights": True,
        },
        "val_size": 0.2,
    },
}

BINS_VALUES = [5, 10, 20]
MASK_PROBS = [0.3, 0.5, 0.8]


def make_config(num_bins, masking_prob):
    name = f"pretrained_baseline_bins{num_bins}_mask{int(masking_prob * 100):03d}"
    notes = (
        f"Pretrained default-scGPT baseline: norm_strategy=binning, num_bins={num_bins}, "
        f"masking_prob={masking_prob}, tasks=[masking, masking_from_cls] (the same 'default "
        f"scGPT' recipe as real_runs/baseline_random_init_default_scgpt.yaml), but with real "
        f"pretraining (max_epochs=200, like real_runs/stage1-4) instead of max_epochs=0, and "
        f"finetune_mode=partial (like real_runs/stage1-4) instead of full. Compare against "
        f"real_runs/baseline_random_init_default_scgpt.yaml's finetune_summary.md (pretrained "
        f"vs. random-init, num_bins=10 cell only) and against "
        f"real_runs/stage4_bins{num_bins}_mask{int(masking_prob * 100):03d}.yaml's "
        f"finetune_summary.md (same grid cell, different task family)."
    )

    cfg = copy.deepcopy(BASE)
    cfg["paths"]["output_dir"] = f"outputs/pretrain/pretrained_baseline/{name}"
    cfg["wandb"]["run_name"] = name
    cfg["wandb"]["run_notes"] = notes
    cfg["training"]["notes"] = notes
    cfg["training"]["masking_prob"] = masking_prob
    cfg["data"]["num_bins"] = num_bins
    return name, cfg


def write_config(cfg, filename):
    path = os.path.join(SWEEP_DIR, filename)
    OmegaConf.save(OmegaConf.create(cfg), path)
    return path


def write_list(paths, filename):
    path = os.path.join(SWEEP_DIR, filename)
    with open(path, "w") as f:
        for p in paths:
            f.write(f"configs/pretrain/pretrained_baseline/{p}\n")
    return path


def generate():
    files = []
    for num_bins in BINS_VALUES:
        for masking_prob in MASK_PROBS:
            name, cfg = make_config(num_bins, masking_prob)
            filename = f"{name}.yaml"
            write_config(cfg, filename)
            files.append(filename)
    write_list(files, "config_list.txt")
    print(f"Pretrained baseline sweep: wrote {len(files)} configs")


if __name__ == "__main__":
    generate()
