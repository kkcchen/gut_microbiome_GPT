# Data-scaling pretraining experiment

Does pretraining a scGPT recipe on **more** of `pretrain.h5ad` (106,398 samples,
431 BioProjects) improve the downstream representation? Run for the two winning
recipes — `baseline_bins20_mask030` and `stage3_concatenation` — at four
training-data fractions (10 / 20 / 50 / 100 %).

## Pipeline

| step | artifact |
|---|---|
| 1. split indices | `../../../scaling_splits/*.json` — per-ratio `{train, val}` positional-index lists into `pretrain.h5ad`, consumed by `utils/data_pipeline.load_scaling_split` via the `data.scaling_split_file` / `data.scaling_split_key` config fields (bypasses the normal `split_data`). |
| 2. configs | `generate_scaling_configs.py` → `scaling_<recipe>_<ratio>.yaml` (+ `scaling_config_list.txt`). Each is the base recipe config with `paths.output_dir`, wandb, and the two `scaling_split_*` keys overridden. |
| 3. pretrain | `sbatch/refactored_pretrain_scaling.sbatch` — SLURM array `0-7`, one config per line of `scaling_config_list.txt`. |
| 4. finetune | `sbatch/finetune_scaling.sbatch` — array `0-19`, 10 checkpoints (8 scaling + 2 July refs) × {HMC 13 tasks, AGP 18 tasks}. `FT_MODE={partial,full}` env selects frozen-encoder linear probe vs end-to-end. Writes `outputs/pretrain/scaling/<FT_MODE>[_agp]/<name>/`. |
| 5. combine | `scripts/combine_scaling_summaries.py` → `outputs/pretrain/scaling/<recipe>_scaling_combined[_full].md` — ratios side by side, 5-fold-CV `mean ± 95% CI`, a `100%-ref` column, `Δ(100%−10%)` / `Δ(ref−100%)`. |
| 6. refresh | `sbatch/regen_scaling_combined.sbatch` — re-runs step 5 (submit `--dependency=afterany:<finetune job>`). |

## v1 caveats (known, not fixed)

The `scaling_manifest_*.json` were authored upstream and are **not** ideal for a
scaling study:

- **val grows with the ratio** (1,064 → 10,640 rows) — each ratio's best checkpoint is
  selected on a different val set;
- **train subsets are not nested** and nearly disjoint (`t_r010 ∩ t_r100` = 45 rows);
- **random** sample-level 90/10 split, not the project-standard **study-isolated** holdout;
- `batch_vocab` (distinct `study_id` in the train subset + 1) drifts 419→432 across
  ratios, and the July reference checkpoints have 410 — the batch-embedding table is a
  different size in every run.

The July reference checkpoints are additionally **less converged** (best epoch 71 vs
120–182) and their original `finetune_summary.md` used the pre-`aee4f67` single-split
downstream eval.

**v2** rebuilds the splits to fix all of this — fixed study-isolated val, nested
stratified-by-study train subsets (constant `batch_vocab`), re-pretrain, full finetuning.
See `scripts/make_study_scaling_manifests.py` and `generate_study_scaling_configs.py`.
