"""
Combine the v2 (study-isolated split) data-scaling results into one markdown file
per recipe: the four ratios side by side with pooled 5-fold-CV `mean ± 95% CI`,
plus a `July (legacy)` column for context.

    python scripts/combine_study_scaling.py

Reads   outputs/pretrain/scaling_v2/full{,_agp}/scaling_<label>_<ratio>/
Writes  outputs/pretrain/scaling_v2/<label>_combined.md

v2 regime: fixed study-isolated validation holdout (same whole BioProjects for every
ratio), nested stratified-by-study training subsets (every study present at every
ratio -> BatchVocabulary / batch-embedding table identical across all 8 runs), FULL
finetuning (whole transformer unfrozen).
"""
import math
import os

from combine_scaling_summaries import (  # low-level parsers, reused
    AGP_TASKS, HMC_TASKS, REG_TASKS, _col_from, fnum, parse_md,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V2 = os.path.join(ROOT, "outputs/pretrain/scaling_v2")

LABELS = ["baseline_bins20_mask030", "stage3_concatenation"]
RATIOS = ["r010", "r020", "r050", "r100"]
JULY = "july"
COL_KEYS = RATIOS + [JULY]
COL_LABEL = {"r010": "10%", "r020": "20%", "r050": "50%", "r100": "100%", JULY: "July (legacy)†"}

# n_train per ratio, from scripts/make_study_scaling_manifests.py stdout
RATIO_NTRAIN = {"r010": 9574, "r020": 19152, "r050": 47881, "r100": 95745}

# original July single-split finetune_summary.md (HMC tasks only)
JULY_SUMMARY = {
    "baseline_bins20_mask030":
        os.path.join(ROOT, "outputs/pretrain/pretrained_baseline/"
                     "pretrained_baseline_bins20_mask030/finetune_summary.md"),
    "stage3_concatenation":
        os.path.join(ROOT, "outputs/pretrain/real_runs/stage3/"
                     "stage3_abundance_emb_concatenation/finetune_summary.md"),
}


def july_col(label):
    """{task: {metric: (mean, None)}} from the original single-split summary (HMC only)."""
    md = parse_md(JULY_SUMMARY[label])
    out = {}
    for t in REG_TASKS + HMC_TASKS + AGP_TASKS:
        m = md.get(t, {})
        if (m.get("Status") or "").startswith("OK"):
            out[t] = {k: (fnum(m.get(k)), None)
                      for k in ("Accuracy", "F1 (macro)", "AUROC", "MAE", "R2")}
        else:
            out[t] = {}
    return out


def load_all():
    data = {}
    for label in LABELS:
        data[label] = {}
        for r in RATIOS:
            data[label][r] = _col_from(
                os.path.join(V2, "full", f"scaling_{label}_{r}"),
                os.path.join(V2, "full_agp", f"scaling_{label}_{r}"))
        data[label][JULY] = july_col(label)
    return data


def get(dl, col, task, metric):
    return dl.get(col, {}).get(task, {}).get(metric)


def cell(pair):
    if not pair or pair[0] is None:
        return "–"
    m, ci = pair
    return f"{m:.3f} ± {ci:.3f}" if ci is not None else f"{m:.4f}"


def metric_table(dl, tasks, metric):
    lines = ["| Task | " + " | ".join(COL_LABEL[k] for k in COL_KEYS) + " | Δ(100%−10%) |",
             "|---|" + "---|" * (len(COL_KEYS) + 1)]
    for t in tasks:
        row, means = [t], {}
        for k in COL_KEYS:
            p = get(dl, k, t, metric)
            means[k] = p[0] if p else None
            row.append(cell(p))
        row.append(f"{means['r100'] - means['r010']:+.4f}"
                   if means["r010"] is not None and means["r100"] is not None else "–")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def agg(dl, tasks, col, metric):
    ms, cis = [], []
    for t in tasks:
        p = get(dl, col, t, metric)
        if p and p[0] is not None:
            ms.append(p[0])
            cis.append(p[1] or 0.0)
    if not ms:
        return None, None
    ci = math.sqrt(sum(c * c for c in cis)) / len(ms) if any(cis) else None
    return sum(ms) / len(ms), ci


def scaling_section(dl):
    out = ["## Scaling comparison\n",
           "Mean across tasks (OK runs only) per training-data fraction; ± = pooled 95% CI "
           "from the per-task 5-fold error bars. A Δ inside the error band is noise.\n"]
    for name, tasks in [("HMC classification", HMC_TASKS),
                        ("AGP classification", AGP_TASKS),
                        ("All classification", HMC_TASKS + AGP_TASKS)]:
        out.append(f"### {name}\n")
        hdr = [f"{COL_LABEL[r]} (n={RATIO_NTRAIN[r]:,})" for r in RATIOS] + [COL_LABEL[JULY]]
        out.append("| Metric | " + " | ".join(hdr) + " | Δ(100%−10%) |")
        out.append("|---|" + "---|" * (len(COL_KEYS) + 1))
        for metric in ["Accuracy", "F1 (macro)", "AUROC"]:
            cells, ms = [metric], {}
            for k in COL_KEYS:
                m, ci = agg(dl, tasks, k, metric)
                ms[k] = m
                cells.append(f"{m:.3f} ± {ci:.3f}" if m is not None and ci is not None
                             else (f"{m:.4f}" if m is not None else "–"))
            cells.append(f"{ms['r100'] - ms['r010']:+.4f}"
                         if ms["r010"] is not None and ms["r100"] is not None else "–")
            out.append("| " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out)


def build(label, dl):
    return "\n".join([
        f"# Data-scaling v2 — `{label}` (study-isolated splits, full finetuning)\n",
        "Pretrained on nested **stratified-by-study** subsets of a fixed **study-isolated** "
        "train pool (`scaling_splits/study_manifest_*.json`); the validation holdout is the "
        "same whole BioProjects for every ratio, and every training study appears at every "
        "ratio so `batch_vocab` (the batch-embedding table) is identical across all runs. "
        "Downstream: **full** finetuning (whole transformer unfrozen), pooled 5-fold CV "
        "(`mean ± 95% CI`); `sex`/`location`/`age`/`bmi` are single cross-study presplit "
        "(bare mean).\n",
        "Subset sizes: " + ", ".join(
            f"{COL_LABEL[r]} = {RATIO_NTRAIN[r]:,}" for r in RATIOS) + " training samples.\n",
        "† `July (legacy)` = the original `pretrained_baseline_bins20_mask030` / "
        "`stage3_abundance_emb_concatenation` checkpoints' downstream numbers, from their "
        "**single fixed train/val split** (pre-`aee4f67`, no CV, no error bar), best "
        "pretrain epoch 71, `batch_vocab` 410, random 90/10 pretrain split. Shown for "
        "context only — **not comparable** to the CV columns. AGP tasks did not exist "
        "for those runs (`–`).\n",
        "---\n",
        scaling_section(dl),
        "---\n",
        "## Per-task — AUROC\n",
        "### HMC tasks\n", metric_table(dl, HMC_TASKS, "AUROC"), "",
        "### AGP tasks\n", metric_table(dl, AGP_TASKS, "AUROC"), "",
        "## Per-task — F1 (macro)\n",
        "### HMC tasks\n", metric_table(dl, HMC_TASKS, "F1 (macro)"), "",
        "### AGP tasks\n", metric_table(dl, AGP_TASKS, "F1 (macro)"), "",
        "## Per-task — Accuracy\n",
        "### HMC tasks\n", metric_table(dl, HMC_TASKS, "Accuracy"), "",
        "### AGP tasks\n", metric_table(dl, AGP_TASKS, "Accuracy"), "",
        "## Regression (age, bmi) — reference only\n",
        "### MAE\n", metric_table(dl, REG_TASKS, "MAE"), "",
        "### R2\n", metric_table(dl, REG_TASKS, "R2"), "",
    ])


def main():
    if not os.path.isdir(os.path.join(V2, "full")) and not os.path.isdir(os.path.join(V2, "full_agp")):
        print(f"no v2 results under {V2}/full[_agp] yet — nothing to write")
        return
    data = load_all()
    os.makedirs(V2, exist_ok=True)
    for label in LABELS:
        path = os.path.join(V2, f"{label}_combined.md")
        with open(path, "w") as f:
            f.write(build(label, data[label]))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
