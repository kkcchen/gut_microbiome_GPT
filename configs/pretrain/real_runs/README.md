# Pretraining sweep — file structure, stages, and how to run

This directory holds a **staged hyperparameter sweep** for pretraining. Each stage
sweeps one group of related parameters while holding everything else fixed, so the
search stays tractable instead of crossing every parameter with every other one.

Every pretrain config anywhere in `configs/pretrain/` (not just this sweep) now
automatically finetunes on 12 downstream tasks after pretraining finishes, and
writes a summary — see "What happens when you run a config" below.

## Directory contents

```
configs/pretrain/
├── debug.yaml, binary.yaml,                     <- standalone base configs, not part
│   taxa_emb_config.yaml                            of this sweep, run individually
└── real_runs/                                    <- this directory: the staged sweep
    ├── README.md                                     (this file)
    ├── generate_sweep_configs.py                      regenerates stage2-4 below
    ├── stage1_norm_*.yaml (8 files)                    Stage 1 configs
    ├── stage1_config_list.txt                          Stage 1 run list
    ├── stage2_*.yaml (12 files)                        Stage 2 configs
    ├── stage2_config_list.txt                          Stage 2 run list
    ├── stage3_*.yaml (5 files)                         Stage 3 configs
    ├── stage3_config_list.txt                          Stage 3 run list
    ├── stage4_*.yaml (9 files)                         Stage 4 configs
    ├── stage4_config_list.txt                          Stage 4 run list
    ├── baseline_random_init_*.yaml (2 files)           Baseline configs (see below)
    └── baseline_config_list.txt                        Baseline run list
```

Each `stageN_config_list.txt` lists that stage's config paths, one per line — the
matching `sbatch/refactored_pretrain_stageN.sbatch` script reads one line per SLURM
array task, so array index 0 runs the first line, index 1 the second, etc.

## What happens when you run a config

When you call the sbatch file, `sbatch sbatch/refactored_pretrain_stageN.sbatch` where N is 1 to 4, this will do a hyperparameter sweep. Each configuration will, in one job,
automatically, for any pretrain config (this sweep or otherwise):

1. Pretrains the model per the config's `training`/`model`/`data` sections.
2. Immediately finetunes the resulting checkpoint on all 12 tasks in
   `configs/finetune/task_registry.yaml` (age, bmi, sex, supplement, and 8 more —
   see that file), using the `finetune:` block in the pretrain config for base
   hyperparameters.
3. Writes/updates `outputs/pretrain/real_runs/<name>/finetune_summary.md` — one
   Markdown table, one row per task, with accuracy/F1/class-weighted-macro-AUROC
   for classification tasks and MAE/RMSE/R² for regression tasks. **This is what
   you read to judge a config**, not just the pretraining validation loss. The
   file is rewritten after *every* task (not only at the end), so if the job
   dies partway through (walltime, OOM), the summary still reflects whatever
   finished — rows for tasks not yet reached show `pending`.

A config **must** have a `finetune:` block, and must run single-GPU/single-process
(the auto-chain deadlocks under multi-GPU DDP) — `scripts/train.py` fails
immediately on either violation, before wasting any compute on pretraining. If any
of the 12 finetune tasks fail, the job still finishes (other tasks aren't blocked)
but `scripts/train.py` exits non-zero so SLURM/`sacct` reflects the partial
failure — check `finetune_summary.md` for which task(s) and why.

## The four stages

| Stage | What's varied | What's held fixed | # configs |
|---|---|---|---|
| **1 — representation** | `data.norm_strategy` (binary, clr, log_rel_abundance, log_counts, rel_abundance, binning) and `data.num_bins` (5, 10, 20 — only under `binning`) | `tasks=["masking"]`, `masking_prob=0.15`, `sample_emb_style=cls`, `abundance_emb_style=continuous`, `use_batch_labels=true` | 8 |
| **2 — task family** | `training.tasks` (`masking`, `masking_from_cls`, `masking_taxa`, and all three combined) × `masking_prob` (0.3, 0.5, 0.8) | Stage 1's winning representation | 12 |
| **3 — architecture** | `sample_emb_style` (cls, avg-pool), `abundance_emb_style` (continuous, scaling, concatenation), `use_batch_labels` (true, false) — each varied **one at a time** against a baseline, not fully crossed | Stage 1 representation + Stage 2 tasks | 5 (1 baseline + 4 variants) |
| **4 — fine sweep** | `num_bins` (5, 10, 20) × `masking_prob` (0.3, 0.5, 0.8), on `norm_strategy=binning` | Stage 3's winning architecture | 9 |

Parameters explicitly **excluded** from all four stages (see the plan for why):
GNN taxa encoder (`use_gnn`, always `false`), taxa embedding initialization
(`preinitialized_taxa_embedding_path` and friends), model capacity
(`d_model`/`nhead`/`d_hid`/`nlayers`/`dropout`, fixed at 128/8/512/3/0.1), optimizer/
schedule (fixed at adamw/cosine/1e-3), and `max_seq_len` (currently a no-op in the
data pipeline regardless of value — don't bother sweeping it).

## Baseline — no-pretraining ablation

`baseline_random_init_winner_arch.yaml` and `baseline_random_init_default_scgpt.yaml`
are **not** part of the Stage 1-4 progression above — they're a fixed, two-config
ablation answering "does pretraining actually help?" by skipping pretraining
entirely (`training.max_epochs: 0`, so the encoder stays randomly initialized) and
using `finetune.training.finetune_mode: "full"` (every layer trains on the
downstream task) instead of the sweep's default `"partial"` (linear probing on a
frozen encoder):

| Config | Architecture | Tasks |
|---|---|---|
| `baseline_random_init_winner_arch` | The sweep's actual winning architecture (`STAGE1/2/3_WINNER` in `generate_sweep_configs.py`): `norm_strategy=log_rel_abundance`, `sample_emb_style=cls`, `abundance_emb_style=concatenation`, `use_batch_labels=true` | `[masking_taxa]`, `masking_prob=0.3` |
| `baseline_random_init_default_scgpt` | A default scGPT-style architecture: `norm_strategy=binning`, `num_bins=20`, `sample_emb_style=cls`, `abundance_emb_style=continuous` (the `../pretrained_baseline/` sweep's winning bins/mask cell, `pretrained_baseline_bins20_mask030`) | `[masking, masking_from_cls]`, `masking_prob=0.3` |

`training.tasks`/`masking_prob` are inert at `max_epochs=0` (no pretraining step
ever runs) — they're kept in the config only for documentation.

`training.max_epochs: 0` is handled specially in `trainers/trainer.py::train()`: the
epoch loop never runs (nothing to train), so instead of falling through with no
checkpoint, it saves the freshly-initialized (random) model weights as the "best"
checkpoint immediately. The auto-finetune chain then picks that up exactly like a
normal pretraining run's checkpoint and finetunes all 12 downstream tasks from
there, `finetune_mode=full` and all.

Read each baseline's `finetune_summary.md` against the *matching-architecture*
stage's own `finetune_summary.md` (Stage 1's `log_rel_abundance` config, or a
`binning`/`masking`+`masking_from_cls` Stage 2 config) to separate two effects at
once: pretrained-vs-random-init, and full-vs-partial finetuning.

## How to run each stage

```bash
sbatch sbatch/refactored_pretrain_stage1.sbatch       # Stage 1 (8 array tasks, 0-7)
sbatch sbatch/refactored_pretrain_stage2.sbatch       # Stage 2 (12 array tasks, 0-11)
sbatch sbatch/refactored_pretrain_stage3.sbatch       # Stage 3 (5 array tasks, 0-4)
sbatch sbatch/refactored_pretrain_stage4.sbatch       # Stage 4 (9 array tasks, 0-8)
sbatch sbatch/refactored_pretrain_baseline.sbatch     # Baseline (2 array tasks, 0-1)
```

Each is a single `sbatch` call — one job array, one array task per config in that
stage's `config_list.txt`. Logs land in `logs/refactored/pretrain/<stage>/` (or
`logs/refactored/pretrain/baseline/` for the baseline).

## How to progress from one stage to the next

**Stages 2, 3, and 4 currently assume a placeholder "winner" from the stage(s)
before them**, because they were generated before Stage 1 actually ran (there was
no real result yet to build on). The placeholders are recorded at the top of
`generate_sweep_configs.py`:

```python
STAGE1_WINNER = {"norm_strategy": "binning", "num_bins": 10}
STAGE2_WINNER = {"tasks": ["masking"], "masking_prob": 0.5}
STAGE3_WINNER = {"sample_emb_style": "cls", "abundance_emb_style": "continuous", "use_batch_labels": True}
```

To progress for real, once a stage's jobs have finished:

1. Look at `outputs/pretrain/real_runs/<run_name>/finetune_summary.md` for every
   config in that stage (or its pretraining validation loss, if you'd rather judge
   by that instead) and decide the actual winner.
2. Edit the matching `STAGEN_WINNER` constant in `generate_sweep_configs.py` to
   match what really won.
3. Regenerate the *next* stage(s) by rerunning:
   ```bash
   python configs/pretrain/real_runs/generate_sweep_configs.py
   ```
   This overwrites all of `stage2_*`, `stage3_*`, `stage4_*` and their
   `config_list.txt` files — Stage 1 itself is never touched by this script (its
   own axis has no upstream winner to depend on).
4. Launch the next stage's `sbatch` command from the table above.

Repeat for each stage in order: Stage 1 → set `STAGE1_WINNER` → regenerate →
Stage 2 → set `STAGE2_WINNER` → regenerate → Stage 3 → set `STAGE3_WINNER` →
regenerate → Stage 4.

The `baseline_random_init_*.yaml` configs are hand-written and standalone, like
Stage 1 — `generate_sweep_configs.py` never touches them, so no regeneration step
is needed for them.

