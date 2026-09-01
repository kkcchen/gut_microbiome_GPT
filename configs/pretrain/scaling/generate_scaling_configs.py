"""
Generator for data-scaling pretraining configs.

For each of the two winning scGPT recipes:
  - pretrained_baseline_bins20_mask030  (binning, num_bins=20, masking_prob=0.3)
  - stage3_abundance_emb_concatenation  (log_rel_abundance, concatenation, masking_taxa)
pretrain on a partial subset of the training data. The train/val row indices for
each subset size come from the scaling manifests in scaling_splits/, consumed by
utils/data_pipeline.load_scaling_split via the new data.scaling_split_file /
data.scaling_split_key config fields (these bypass the study_id group split).

    python configs/pretrain/scaling/generate_scaling_configs.py

Writes scaling_<label>_<ratio>.yaml + scaling_config_list.txt in this directory.
"""
import copy
import os

from omegaconf import OmegaConf

SCALING_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCALING_DIR, "..", "..", ".."))

# (label, source config relative to repo root)
BASE_CONFIGS = [
    ("baseline_bins20_mask030",
     "configs/pretrain/pretrained_baseline/pretrained_baseline_bins20_mask030.yaml"),
    ("stage3_concatenation",
     "configs/pretrain/real_runs/stage3_abundance_emb_concatenation.yaml"),
]

# manifest file (in scaling_splits/) and the split key inside it
RATIOS = [
    ("r010", "scaling_splits/scaling_manifest_r010.json", "r010"),
    ("r020", "scaling_splits/scaling_manifest_r020.json", "r020"),
    ("r050", "scaling_splits/scaling_manifest_r050.json", "r050"),
    ("r100", "scaling_splits/scaling_manifest_r100.json", "r100"),
]

WANDB_PROJECT = "microbiome-pretrain-scaling"


def generate():
    files = []
    for label, src in BASE_CONFIGS:
        base = OmegaConf.load(os.path.join(REPO_ROOT, src))
        for ratio_slug, manifest, split_key in RATIOS:
            cfg = copy.deepcopy(base)
            name = f"scaling_{label}_{ratio_slug}"
            cfg.paths.output_dir = f"outputs/pretrain/scaling/{name}"
            cfg.data.scaling_split_file = manifest
            cfg.data.scaling_split_key = split_key
            notes = (
                f"Data-scaling run: {label} recipe pretrained on the {ratio_slug} "
                f"subset ({manifest}, key={split_key})."
            )
            cfg.wandb.project = WANDB_PROJECT
            cfg.wandb.run_name = name
            cfg.wandb.run_notes = notes
            cfg.training.notes = notes

            filename = f"{name}.yaml"
            OmegaConf.save(cfg, os.path.join(SCALING_DIR, filename))
            files.append(filename)

    list_path = os.path.join(SCALING_DIR, "scaling_config_list.txt")
    with open(list_path, "w") as f:
        for p in files:
            f.write(f"configs/pretrain/scaling/{p}\n")
    print(f"Wrote {len(files)} configs + scaling_config_list.txt")


if __name__ == "__main__":
    generate()
