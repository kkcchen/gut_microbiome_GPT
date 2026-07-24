# Raw baselines: XGBoost (grid) + ElasticNet (grid) directly on raw features, 4 normalizations

This directory holds eval configs that run classical ML baselines directly
on RAW (un-embedded) taxa abundance data -- no trained model, no forward
pass (`eval.with_model: False`) -- as a floor/control comparison against
the model-embedding baselines in `../embedding_baselines/`.

For each of 4 normalization strategies applied to the raw AnnData features
(`utils/downstream_data_utils.py::normalize_embeddings`), a single
`accelerate launch -m scripts.eval_downstream_only --config <path>` run
trains/evaluates:
  - `xgboost` with the same grid search used everywhere else in the repo
    (`utils/downstream_models_utils.py::train_xgboost`, `search_type="grid"`,
    72 combos, cv=3)
  - `linear` (ElasticNet / Elastic-Net Logistic Regression), a new small
    grid (`utils/downstream_models_utils.py::train_linear`, `search_type="grid"`,
    15 combos, cv=3)

on all 11 classification tasks in `downstream_tasks_config.tasks` (same set
as `../embedding_baselines/`, age/bmi regression tasks excluded).

## Normalizations

| Config | `downstream_tasks_config.normalization` | Formula |
|---|---|---|
| `rel_ab.yaml` | `rel_ab` | `closure(x + 1e-8)` |
| `log_rel_abundance.yaml` | `log_rel_abundance` | `log(closure(x + 1e-8))` -- matches the pretrain-side winning strategy (`data_utils/collator.py::apply_normalization`) |
| `clr.yaml` | `clr` | `clr(closure(x + 1e-8))` |
| `none.yaml` | `none` | unnormalized raw counts (floor control) |

## Regenerating configs

```bash
python configs/eval/raw_baselines/generate_configs.py
```

Overwrites all 4 YAMLs and `config_list.txt`. Edit `NORMALIZATIONS`/`BASE`
in `generate_configs.py` to add a normalization or change methods/tasks.

## How to run

```bash
sbatch sbatch/eval_raw_baselines.sbatch   # 4 array tasks, 0-3
```

or locally/foreground for a single config:

```bash
accelerate launch --main_process_port 0 -m scripts.eval_downstream_only \
  --config configs/eval/raw_baselines/log_rel_abundance.yaml
```

Logs land in `logs/refactored/eval/raw_baselines/`.

## Relationship to `configs/eval/raw.yaml`

`../raw.yaml` remains the original single hand-maintained raw-baseline
example (CLR only, `xgboost.search_type: "none"`, fixed params, points at
`/project/aip-rahulgk/...` paths). This directory is the new grid-search
sweep and uses the accessible `/scratch/kchen13/hmc_final_fixed/...` data
paths instead.

## Reading results

Each config run writes raw per-(task, method) metrics under
`outputs/eval/raw_baselines/<normalization>/downstream_tasks/<task>/<method>/`,
but you shouldn't need to read those directly. At the end of every run,
`utils/downstream_summary_utils.py::write_downstream_summary` compiles them
into:

- `outputs/eval/raw_baselines/<normalization>/summary.md` -- one Markdown
  table per method (xgboost, linear) for that normalization, one row per task.
- `outputs/eval/raw_baselines/combined_summary.md` -- refreshed after every
  run, one Markdown table per method with one row per (normalization, task)
  across all sibling configs in this directory. This is the single file to
  check for comparing normalizations against each other, and against
  `outputs/eval/embedding_baselines/combined_summary.md`.
