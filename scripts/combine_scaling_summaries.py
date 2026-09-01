"""
Combine the per-run HMC + AGP finetune summaries for the data-scaling experiments
into ONE markdown file per pretrain recipe (two files total), with the scaling
ratios (plus a re-evaluated 100%-ref column) side by side, 5-fold CV error bars,
and a scaling-comparison section.

    python scripts/combine_scaling_summaries.py

Reads, per recipe <label> and ratio <ratio> in {r010,r020,r050,r100}:
  outputs/pretrain/scaling/scaling_<label>_<ratio>/finetune/<task>/cv_summary.yaml   (HMC, 5-fold)
  outputs/pretrain/scaling/scaling_<label>_<ratio>/finetune_summary.md               (HMC, presplit fallback: sex/location + regression)
  outputs/pretrain/scaling/finetune_agp/scaling_<label>_<ratio>/finetune/<task>/cv_summary.yaml  (AGP, 5-fold)
and the same layout under ref100/<label>/ + ref100_agp/<label>/ for the 100%-ref column
(the original July checkpoints re-finetuned through the identical CV -- see footnote).
Writes:
  outputs/pretrain/scaling/<label>_scaling_combined.md
"""
import glob
import math
import os

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCALING = os.path.join(ROOT, "outputs/pretrain/scaling")

LABELS = ["baseline_bins20_mask030", "stage3_concatenation"]
RATIOS = ["r010", "r020", "r050", "r100"]
REF = "ref"  # original July checkpoint, re-finetuned through the same 5-fold CV
COL_KEYS = RATIOS + [REF]
RATIO_PCT = {"r010": "10%", "r020": "20%", "r050": "50%", "r100": "100%", REF: "100%-ref†"}

# n_train per ratio (from scaling_splits/scaling_manifest_*.json)
RATIO_NTRAIN = {"r010": 9576, "r020": 19152, "r050": 47879, "r100": 95758}

HMC_TASKS = [
    "sex", "supplement", "helicobacter_pylori_infection", "hbv_infection",
    "hiv_infection", "sunflower_seed_oil_emollient_therapy", "chemotherapy",
    "type_1_diabetes", "crohns_disease", "diarrhea", "location",
]
REG_TASKS = ["age", "bmi"]
AGP_TASKS = [
    "age cat", "alcohol_frequency", "antibiotic_history", "asd", "autoimmune",
    "contraceptive", "diabetes", "fruit_frequency", "gluten", "ibd", "ibs",
    "meat_eggs_frequency", "probiotic_frequency", "sibo", "sleep_duration",
    "sugary_sweets_frequency", "vegetable_frequency", "weight_change",
]

# Metric -> ordered list of cv_summary aggregate keys to try (first present wins).
CV_METRIC_KEYS = {
    "Accuracy": ["test_accuracy"],
    "F1 (macro)": ["test_f1_macro", "test_f1"],
    "AUROC": ["test_auroc_weighted", "test_auroc"],
}
# Columns of the markdown finetune_summary.md table (for presplit / regression fallback).
MD_COLS = ["Type", "Status", "Accuracy", "F1 (macro)", "F1 (weighted)", "AUROC", "MAE", "RMSE", "R2"]


def fnum(v):
    if v is None:
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def parse_md(path):
    """Fallback: {task: {md_col: str_or_None}} from a finetune_summary.md table."""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            if not line.startswith("|") or line.startswith("| Task") or set(line.strip()) <= set("|- "):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) < 10:
                continue
            out[cells[0]] = {c: (None if v in ("–", "-", "") else v)
                             for c, v in zip(MD_COLS, cells[1:10])}
    return out


def parse_cv(run_dir):
    """{task: {metric: (mean, ci_margin)}} from run_dir/finetune/<task>/cv_summary.yaml."""
    out = {}
    for path in glob.glob(os.path.join(run_dir, "finetune", "*", "cv_summary.yaml")):
        task = os.path.basename(os.path.dirname(path))
        try:
            agg = (yaml.safe_load(open(path)) or {}).get("aggregates", {})
        except Exception:
            continue
        tvals = {}
        for metric, keys in CV_METRIC_KEYS.items():
            for k in keys:
                if k in agg:
                    tvals[metric] = (fnum(agg[k].get("mean")), fnum(agg[k].get("ci_margin")))
                    break
        if tvals:
            out[task] = tvals
    return out


def _col_from(hmc_dir, agp_dir):
    """Merge one column: 5-fold CV where available, else the md mean (ci None).

    md fallback is taken from the run that actually owns the task (HMC run for
    HMC/regression tasks, AGP run for AGP tasks) -- the other run lists it as SKIPPED.
    """
    cv = {**parse_cv(hmc_dir), **parse_cv(agp_dir)}
    md_hmc = parse_md(os.path.join(hmc_dir, "finetune_summary.md"))
    md_agp = parse_md(os.path.join(agp_dir, "finetune_summary.md"))
    md_for = {t: md_hmc for t in REG_TASKS + HMC_TASKS}
    md_for.update({t: md_agp for t in AGP_TASKS})
    merged = {}
    for t in REG_TASKS + HMC_TASKS + AGP_TASKS:
        m = md_for[t].get(t, {})
        if t in cv:
            merged[t] = cv[t]
        elif (m.get("Status") or "").startswith("OK"):
            merged[t] = {k: (fnum(m.get(k)), None) for k in ("Accuracy", "F1 (macro)", "AUROC", "MAE", "R2")}
        else:
            merged[t] = {}
    return merged


# Where each finetune mode reads its HMC / AGP per-run summaries from, relative to SCALING.
#   fn(label, ratio_or_None) -> ordered list of (hmc_dir, agp_dir) candidates;
#   the first pair with an existing dir wins. Primary = the consolidated
#   sbatch/finetune_scaling.sbatch layout (<mode>/<name>, <mode>_agp/<name>);
#   fallback = the v1 one-off layout (inline HMC, finetune_agp/, ref100/).
def _mode_dirs(mode, label, ratio):
    name = f"scaling_{label}_{ratio}" if ratio else label
    cands = [(os.path.join(SCALING, mode, name),
              os.path.join(SCALING, f"{mode}_agp", name))]
    if ratio:
        cands.append((os.path.join(SCALING, name),
                      os.path.join(SCALING, "finetune_agp", name)))
    else:
        cands.append((os.path.join(SCALING, "ref100", label),
                      os.path.join(SCALING, "ref100_agp", label)))
    return cands


MODES = ["partial", "full"]


def _resolve(mode, label, ratio):
    """First (hmc, agp) candidate where either dir exists; else the primary pair."""
    cands = _mode_dirs(mode, label, ratio)
    for hmc, agp in cands:
        if os.path.isdir(hmc) or os.path.isdir(agp):
            return hmc, agp
    return cands[0]


def load_all(mode="partial"):
    """data[label][col][task] = {metric: (mean, ci_or_None)}"""
    data = {}
    for label in LABELS:
        data[label] = {}
        for ratio in RATIOS:
            data[label][ratio] = _col_from(*_resolve(mode, label, ratio))
        data[label][REF] = _col_from(*_resolve(mode, label, None))
    return data


def has_any(mode):
    """True if at least one run directory for this mode exists (so it's worth writing)."""
    for label in LABELS:
        for ratio in list(RATIOS) + [None]:
            for d in _resolve(mode, label, ratio):
                if os.path.isdir(d):
                    return True
    return False


def cell(pair):
    """(mean, ci) -> 'm ± ci' / 'm' / '–'."""
    if pair is None:
        return "–"
    m, ci = pair
    if m is None:
        return "–"
    return f"{m:.3f} ± {ci:.3f}" if ci is not None else f"{m:.4f}"


def get(data_label, col, task, metric):
    return data_label.get(col, {}).get(task, {}).get(metric)


def metric_table(data_label, tasks, metric):
    hdr = [RATIO_PCT[k] for k in COL_KEYS]
    lines = ["| Task | " + " | ".join(hdr) + " | Δ(100%−10%) | Δ(ref−100%) |",
             "|---|" + "---|" * (len(COL_KEYS) + 2)]
    for t in tasks:
        row = [t]
        means = {}
        for k in COL_KEYS:
            p = get(data_label, k, t, metric)
            means[k] = p[0] if p else None
            row.append(cell(p))
        row.append(f"{means['r100'] - means['r010']:+.4f}"
                   if means["r010"] is not None and means["r100"] is not None else "–")
        row.append(f"{means[REF] - means['r100']:+.4f}"
                   if means[REF] is not None and means["r100"] is not None else "–")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def agg_mean(data_label, tasks, col, metric):
    """Mean of per-task means + pooled ci (rms of per-task ci / sqrt(n))."""
    ms, cis = [], []
    for t in tasks:
        p = get(data_label, col, t, metric)
        if p and p[0] is not None:
            ms.append(p[0])
            cis.append(p[1] if p[1] is not None else 0.0)
    if not ms:
        return None, None
    mean = sum(ms) / len(ms)
    pooled_ci = math.sqrt(sum(c * c for c in cis)) / len(ms) if any(cis) else None
    return mean, pooled_ci


def scaling_section(data_label):
    lines = ["## Scaling comparison\n",
             "Mean across tasks (OK runs only) per training-data fraction. "
             "± is the pooled 95% CI from the per-task 5-fold error bars; a Δ smaller "
             "than the two cells' error bars is within noise.\n"]
    for group, tasks in [("HMC classification", HMC_TASKS),
                         ("AGP classification", AGP_TASKS),
                         ("All classification", HMC_TASKS + AGP_TASKS)]:
        lines.append(f"### {group}\n")
        hdr = [f"{RATIO_PCT[r]} (n={RATIO_NTRAIN[r]:,})" for r in RATIOS] + [RATIO_PCT[REF]]
        lines.append("| Metric | " + " | ".join(hdr) + " | Δ(100%−10%) | Δ(ref−100%) |")
        lines.append("|---|" + "---|" * (len(COL_KEYS) + 2))
        for metric in ["Accuracy", "F1 (macro)", "AUROC"]:
            cells = [metric]
            ms = {}
            for k in COL_KEYS:
                m, ci = agg_mean(data_label, tasks, k, metric)
                ms[k] = m
                cells.append(f"{m:.3f} ± {ci:.3f}" if m is not None and ci is not None
                             else (f"{m:.4f}" if m is not None else "–"))
            cells.append(f"{ms['r100'] - ms['r010']:+.4f}"
                         if ms["r010"] is not None and ms["r100"] is not None else "–")
            cells.append(f"{ms[REF] - ms['r100']:+.4f}"
                         if ms[REF] is not None and ms["r100"] is not None else "–")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


FT_MODE_BLURB = {
    "partial": "**Finetune mode: partial** — the pretrained transformer is frozen; only a "
               "2-layer MLP head trains (linear-probe-style readout of the frozen representation).",
    "full": "**Finetune mode: full** — the whole pretrained transformer is unfrozen and "
            "trained end-to-end with the head (`finetune.training.finetune_mode=full`). Same "
            "pooled 5-fold CV, seed, folds and base h5ads as the partial run.",
}


def build(label, data_label, mode="partial"):
    L = [f"# Data-scaling finetune results — `{label}`"
         + ("" if mode == "partial" else "  (full finetuning)") + "\n",
         FT_MODE_BLURB[mode] + "\n",
         "Pretrained on partial subsets of the training data (train/val indices from "
         "`scaling_splits/scaling_manifest_*.json`), then finetuned on all downstream "
         "tasks. Single-study tasks: pooled 5-fold CV (`mean ± 95% CI`). Multi-study "
         "`sex`/`location` and regression `age`/`bmi`: single presplit, no error bar.\n",
         "Subset sizes: " + ", ".join(
             f"{RATIO_PCT[r]} = {RATIO_NTRAIN[r]:,} train samples" for r in RATIOS) + ".\n",
         "† `100%-ref` = the original winning checkpoint (`pretrained_baseline_bins20_mask030` "
         "/ `stage3_abundance_emb_concatenation`), pretrained on the **study-grouped** split "
         "(not a scaling manifest), re-finetuned here through the **identical** 5-fold CV "
         "(same seed, same base h5ads) as every scaling column — so the *evaluation* is "
         "directly comparable. It is NOT a controlled ablation: those checkpoints were "
         "pretrained with a study-grouped 90/10 split and `batch_vocab`=410, vs the "
         "manifests' random 90/10 split and `batch_vocab` 419/425/430/432. `Δ(ref−100%)` "
         "mixes the pretrain-split difference with that vocab difference. Also note each "
         "scaling ratio's best checkpoint is selected on a *different, nested* manifest val "
         "set (1,064 / 2,128 / 5,320 / 10,640 rows).\n",
         "---\n",
         scaling_section(data_label),
         "---\n",
         "## Per-task — AUROC\n",
         "### HMC tasks\n", metric_table(data_label, HMC_TASKS, "AUROC"), "",
         "### AGP tasks\n", metric_table(data_label, AGP_TASKS, "AUROC"), "",
         "## Per-task — F1 (macro)\n",
         "### HMC tasks\n", metric_table(data_label, HMC_TASKS, "F1 (macro)"), "",
         "### AGP tasks\n", metric_table(data_label, AGP_TASKS, "F1 (macro)"), "",
         "## Per-task — Accuracy\n",
         "### HMC tasks\n", metric_table(data_label, HMC_TASKS, "Accuracy"), "",
         "### AGP tasks\n", metric_table(data_label, AGP_TASKS, "Accuracy"), "",
         "## Regression (age, bmi) — reference only, not used for model selection\n",
         "### MAE\n", metric_table(data_label, REG_TASKS, "MAE"), "",
         "### R2\n", metric_table(data_label, REG_TASKS, "R2"), "",
         ]
    return "\n".join(L)


def main():
    modes = ["partial"] + (["full"] if has_any("full") else [])
    for mode in modes:
        data = load_all(mode)
        suffix = "" if mode == "partial" else "_full"
        for label in LABELS:
            out_path = os.path.join(SCALING, f"{label}_scaling_combined{suffix}.md")
            with open(out_path, "w") as f:
                f.write(build(label, data[label], mode))
            print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
