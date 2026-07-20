# Pretrained default-scGPT baseline sweep

This directory holds a 9-config sweep answering: "if the default scGPT-style
architecture in `../real_runs/baseline_random_init_default_scgpt.yaml` actually
pretrains (instead of skipping straight to a random-init encoder), how well does
it do on the 12 downstream tasks, across binning granularity and masking
probability?"

## How this differs from other configs in `configs/pretrain/`

| | `real_runs/baseline_random_init_default_scgpt.yaml` | `real_runs/stage4_bins{N}_mask{P}.yaml` | This sweep |
|---|---|---|---|
| Architecture | default scGPT (binning, num_bins=20, cls, continuous, use_batch_labels) | Stage 1-3 winners | Same default scGPT architecture as the baseline |
| `training.max_epochs` | 0 (random-init, no pretraining) | 200 | 200 |
| `training.tasks` | `[masking, masking_from_cls]`, `masking_prob=0.3` (inert at max_epochs=0) | Stage 2's winning task | `[masking, masking_from_cls]` (fixed, actually active) |
| `data.num_bins` | 20 only | swept: 5, 10, 20 | swept: 5, 10, 20 |
| `training.masking_prob` | 0.3 only (inert) | swept: 0.3, 0.5, 0.8 | swept: 0.3, 0.5, 0.8 |
| `finetune.training.finetune_mode` | `full` | `partial` | `partial` |

So this sweep sits between the two: same architecture and task family as the
baseline, but with real pretraining, `partial` finetuning, and a
`num_bins`x`masking_prob` grid shaped exactly like Stage 4's.

## What happens when you run a config

Same auto-chain as every other pretrain config in this repo (see
`../real_runs/README.md` for the full explanation): pretrain, then
finetune on all 12 tasks in `configs/finetune/task_registry.yaml`, then
write/update `outputs/pretrain/pretrained_baseline/<name>/finetune_summary.md`
after every task.

## Regenerating configs

```bash
python configs/pretrain/pretrained_baseline/generate_configs.py
```

Overwrites all 9 `pretrained_baseline_bins{N}_mask{P}.yaml` files and
`config_list.txt`. Edit `BINS_VALUES`/`MASK_PROBS`/`BASE` in
`generate_configs.py` to change the sweep.

## How to run

```bash
sbatch sbatch/refactored_pretrain_pretrained_baseline.sbatch   # 9 array tasks, 0-8
```

Logs land in `logs/refactored/pretrain/pretrained_baseline/`.

## Reading results

Read each `outputs/pretrain/pretrained_baseline/<name>/finetune_summary.md`
against:
- `outputs/pretrain/real_runs/baseline/baseline_random_init_default_scgpt/finetune_summary.md`
  (num_bins=20 cell only) to separate pretrained-vs-random-init.
- `outputs/pretrain/real_runs/stage4/stage4_bins{N}_mask{P}/finetune_summary.md`
  (same grid cell) to separate the effect of task family
  (`[masking, masking_from_cls]` here vs. Stage 2's winning task there).
