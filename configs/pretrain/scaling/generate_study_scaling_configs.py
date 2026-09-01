"""
Generator for the v2 data-scaling pretraining configs (study-isolated splits).

Same two recipes as generate_scaling_configs.py -- baseline_bins20_mask030 and
stage3_concatenation -- but pointed at the v2 manifests from
scripts/make_study_scaling_manifests.py: a fixed study-isolated validation holdout
shared by every ratio, and nested stratified-by-study training subsets (so the
BatchVocabulary / batch-embedding table is identical across all 8 runs).

    python configs/pretrain/scaling/generate_study_scaling_configs.py

Writes scaling_v2_<label>_<ratio>.yaml + scaling_v2_config_list.txt in this directory.
Outputs go to outputs/pretrain/scaling_v2/ (a fresh tree -- v1 results untouched, and
scripts/train.py's skip-guard never fires).
"""
import copy
import os

from omegaconf import OmegaConf

SCALING_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCALING_DIR, "..", "..", ".."))

BASE_CONFIGS = [
    ("baseline_bins20_mask030",
     "configs/pretrain/pretrained_baseline/pretrained_baseline_bins20_mask030.yaml"),
    ("stage3_concatenation",
     "configs/pretrain/real_runs/stage3_abundance_emb_concatenation.yaml"),
]

# (ratio_slug, manifest path relative to repo root, split key inside the manifest)
RATIOS = [
    ("r010", "scaling_splits/study_manifest_r010.json", "r010"),
    ("r020", "scaling_splits/study_manifest_r020.json", "r020"),
    ("r050", "scaling_splits/study_manifest_r050.json", "r050"),
    ("r100", "scaling_splits/study_manifest_r100.json", "r100"),
]

WANDB_PROJECT = "microbiome-pretrain-scaling-v2"


def generate():
    files = []
    for label, src in BASE_CONFIGS:
        base = OmegaConf.load(os.path.join(REPO_ROOT, src))
        for ratio_slug, manifest, split_key in RATIOS:
            cfg = copy.deepcopy(base)
            name = f"scaling_v2_{label}_{ratio_slug}"
            cfg.paths.output_dir = f"outputs/pretrain/scaling_v2/scaling_{label}_{ratio_slug}"
            cfg.data.scaling_split_file = manifest
            cfg.data.scaling_split_key = split_key
            notes = (
                f"Data-scaling v2: {label} pretrained on the {ratio_slug} stratified-by-study "
                f"subset of the study-isolated train pool ({manifest})."
            )
            cfg.wandb.project = WANDB_PROJECT
            cfg.wandb.run_name = name
            cfg.wandb.run_notes = notes
            cfg.training.notes = notes

            filename = f"{name}.yaml"
            OmegaConf.save(cfg, os.path.join(SCALING_DIR, filename))
            files.append(filename)

    list_path = os.path.join(SCALING_DIR, "scaling_v2_config_list.txt")
    with open(list_path, "w") as f:
        for p in files:
            f.write(f"configs/pretrain/scaling/{p}\n")
    print(f"Wrote {len(files)} configs + scaling_v2_config_list.txt")


if __name__ == "__main__":
    generate()
