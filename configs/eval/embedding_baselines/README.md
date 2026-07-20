# Embedding baselines: XGBoost / Random Forest / Linear / TabPFN on the two winning checkpoints

This directory holds eval configs that, for each of the two sweep-winning
pretrained checkpoints, extract that model's output embeddings on the
downstream train/test h5ads and directly train/evaluate classical ML
baselines on those embeddings -- no new code, reusing the existing generic
pipeline (`scripts/eval.py` -> `utils/downstream_utils.py` ->
`utils/downstream_models_utils.py`'s `MODEL_REGISTRY`). See
`configs/eval/downstream_probe_template.yaml` for the template this is based
on.

## What happens when you run a config

A single `python -m scripts.eval --config <path>` run does the whole
pipeline end to end, in one process:
1. Loads the checkpoint (`paths.pretrained_model_dir`/`paths.checkpoint_path`)
   and runs a forward pass over the eval h5ads, producing per-sample `cls`
   embeddings.
2. Saves those embeddings to an intermediate `.h5ad` under `paths.output_dir`.
3. Immediately feeds them into `run_downstream_evaluation()`, which trains
   and evaluates `xgboost`, `random_forest`, `linear`, and `tabpfn` (each with
   `RandomizedSearchCV`) on the embeddings for all 13 tasks in
   `downstream_tasks_config.tasks`, writing metrics under `paths.output_dir`.

## Targets

| Config | Checkpoint | Architecture |
|---|---|---|
| `stage3_abundance_emb_concatenation.yaml` | `outputs/pretrain/real_runs/stage3/stage3_abundance_emb_concatenation` | overall sweep winner: `norm_strategy=log_rel_abundance`, `abundance_emb_style=concatenation`, `sample_emb_style=cls` |
| `pretrained_baseline_bins20_mask030.yaml` | `outputs/pretrain/pretrained_baseline/pretrained_baseline_bins20_mask030` | best default-scGPT-style architecture: `norm_strategy=binning`, `num_bins=20`, `abundance_emb_style=continuous`, `sample_emb_style=cls` |

Each config's `data`/`model.params` fields are copied from that checkpoint's
own pretrain config -- they must match what the model was trained with.

## Regenerating configs

```bash
python configs/eval/embedding_baselines/generate_configs.py
```

Overwrites both YAMLs and `config_list.txt`. Edit `TARGETS`/`BASE` in
`generate_configs.py` to add another checkpoint or change the downstream
tasks/methods.

## How to run

```bash
sbatch sbatch/refactored_eval_embedding_baselines.sbatch   # 2 array tasks, 0-1
```

or locally/foreground for a single config:

```bash
accelerate launch --main_process_port 0 -m scripts.eval --config configs/eval/embedding_baselines/stage3_abundance_emb_concatenation.yaml
```

Logs land in `logs/refactored/eval/embedding_baselines/`.

## Reading results

`outputs/eval/embedding_baselines/<name>/` contains the extracted embeddings
(`downstream_train.h5ad`, `downstream_test.h5ad`) and per-task metrics
written by `evaluate_multiclass_and_save`/`evaluate_regression_and_save`
(accuracy, F1-macro, AUROC for classification; regression metrics + scatter
plot for age/bmi).
