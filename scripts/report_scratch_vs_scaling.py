"""
Report the no-pretraining (random-init) scGPT baseline next to the v2 data-scaling
columns, per recipe.

    python scripts/report_scratch_vs_scaling.py

The "scratch" model = the two committed random-init configs
(`configs/pretrain/real_runs/baseline_random_init_{default_scgpt,winner_arch}.yaml`,
`training.max_epochs=0`, `finetune.training.finetune_mode=full`), finetuned on every
downstream task through the *identical* pooled 5-fold CV harness as the scaling_v2
columns — so the columns are directly comparable.

Recipe map:
    baseline_bins20_mask030  <- baseline_random_init_default_scgpt   (binning, num_bins=20, continuous)
    stage3_concatenation     <- baseline_random_init_winner_arch     (log_rel_abundance, concatenation)

Reads
    <scratch-hmc-root>/<cfg>/finetune/<task>/cv_summary.yaml   (HMC 9 CV tasks)
    <scratch-hmc-root>/<cfg>/finetune_summary.md               (HMC sex/location/regression, bare mean)
    <scratch-agp-root>/<cfg>/finetune/<task>/cv_summary.yaml   (AGP 18 CV tasks)
    outputs/pretrain/scaling_v2/full{,_agp}/scaling_<label>_<ratio>/...   (ratio columns)
Writes
    outputs/pretrain/scaling_v2/scratch_vs_pretrained.md   (--out to override)

`--reconcile HMC_ROOT AGP_ROOT` instead emits a per-task table comparing the default
(July-30) scratch run against a re-run under those roots, flagging tasks whose AUROC
means differ by more than the combined 95% CI.
"""
import argparse
import math
import os

from combine_scaling_summaries import (  # low-level parsers, reused
    AGP_TASKS, HMC_TASKS, REG_TASKS, _col_from, fnum, parse_cv, parse_md,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V2 = os.path.join(ROOT, "outputs/pretrain/scaling_v2")

# recipe label -> random-init config basename
CFG = {
    "baseline_bins20_mask030": "baseline_random_init_default_scgpt",
    "stage3_concatenation": "baseline_random_init_winner_arch",
}
LABELS = list(CFG)
RATIOS = ["r010", "r020", "r050", "r100"]
SCRATCH = "scratch"
COL_KEYS = [SCRATCH] + RATIOS
COL_LABEL = {SCRATCH: "scratch (0%)", "r010": "10%", "r020": "20%", "r050": "50%", "r100": "100%"}
RATIO_NTRAIN = {"r010": 9574, "r020": 19152, "r050": 47881, "r100": 95745}

DEF_SCRATCH_HMC = os.path.join(ROOT, "outputs/pretrain/real_runs/baseline_seeds")
DEF_SCRATCH_AGP = os.path.join(ROOT, "outputs/pretrain/real_runs/baseline_agp")


def scratch_col(label, hmc_root, agp_root):
    cfg = CFG[label]
    return _col_from(os.path.join(hmc_root, cfg), os.path.join(agp_root, cfg))


def load_all(hmc_root, agp_root):
    data = {}
    for label in LABELS:
        d = {SCRATCH: scratch_col(label, hmc_root, agp_root)}
        for r in RATIOS:
            d[r] = _col_from(os.path.join(V2, "full", f"scaling_{label}_{r}"),
                             os.path.join(V2, "full_agp", f"scaling_{label}_{r}"))
        data[label] = d
    return data


def get(dl, col, task, metric):
    return dl.get(col, {}).get(task, {}).get(metric)


def cell(pair):
    if not pair or pair[0] is None:
        return "–"
    m, ci = pair
    return f"{m:.3f} ± {ci:.3f}" if ci is not None else f"{m:.4f}"


def metric_table(dl, tasks, metric):
    lines = ["| Task | " + " | ".join(COL_LABEL[k] for k in COL_KEYS) + " | Δ(pretrain best − scratch) |",
             "|---|" + "---|" * (len(COL_KEYS) + 1)]
    for t in tasks:
        row, means = [t], {}
        for k in COL_KEYS:
            p = get(dl, k, t, metric)
            means[k] = p[0] if p else None
            row.append(cell(p))
        ratio_means = [means[r] for r in RATIOS if means[r] is not None]
        if means[SCRATCH] is not None and ratio_means:
            best = max(ratio_means) if metric != "MAE" else min(ratio_means)
            row.append(f"{best - means[SCRATCH]:+.4f}")
        else:
            row.append("–")
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


def macro_section(dl):
    out = ["## Macro-average (mean across classification tasks)\n",
           "± = pooled 95% CI from the per-task 5-fold error bars. `Δ` = (mean of the four "
           "pretrained ratios) − scratch.\n"]
    for name, tasks in [("HMC classification", HMC_TASKS),
                        ("AGP classification", AGP_TASKS),
                        ("All classification", HMC_TASKS + AGP_TASKS)]:
        out.append(f"### {name}\n")
        hdr = [COL_LABEL[SCRATCH]] + [f"{COL_LABEL[r]} (n={RATIO_NTRAIN[r]:,})" for r in RATIOS]
        out.append("| Metric | " + " | ".join(hdr) + " | Δ(pretrained mean − scratch) |")
        out.append("|---|" + "---|" * (len(COL_KEYS) + 1))
        for metric in ["Accuracy", "F1 (macro)", "AUROC"]:
            cells, ms = [metric], {}
            for k in COL_KEYS:
                m, ci = agg(dl, tasks, k, metric)
                ms[k] = m
                cells.append(f"{m:.3f} ± {ci:.3f}" if m is not None and ci is not None
                             else (f"{m:.4f}" if m is not None else "–"))
            rvals = [ms[r] for r in RATIOS if ms[r] is not None]
            if ms[SCRATCH] is not None and rvals:
                cells.append(f"{sum(rvals) / len(rvals) - ms[SCRATCH]:+.4f}")
            else:
                cells.append("–")
            out.append("| " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out)


def build(label, dl):
    return "\n".join([
        f"# No-pretraining baseline vs data-scaling — `{label}`\n",
        "**scratch (0%)** = random-init model (`" + CFG[label] + ".yaml`, "
        "`training.max_epochs=0`, `finetune.training.finetune_mode=full`), finetuned "
        "directly on each downstream task through the same pooled 5-fold CV harness "
        "(same seed, folds, base h5ads, 10-epoch full finetune) as the scaling columns. "
        "The 10–100% columns are copied from `" + label + "_combined.md` (v2 study-isolated "
        "pretraining subsets). `sex`/`location`/`age`/`bmi` are single cross-study presplit "
        "(bare mean, no error bar) in every column.\n",
        "Architecture is identical across all columns (`d_model 128, nhead 8, d_hid 512, "
        "nlayers 3`); the only thing that changes left-to-right is how much unsupervised "
        "pretraining the encoder saw before finetuning (none → 95,745 samples).\n",
        "---\n",
        macro_section(dl),
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
    ])


def reconcile(hmc_a, agp_a, hmc_b, agp_b):
    """Per-task AUROC: run A (default/July) vs run B (re-run), flag outside combined CI."""
    lines = ["# Scratch re-run reconciliation — AUROC\n",
             f"A = `{hmc_a}` / `{agp_a}`  ·  B = `{hmc_b}` / `{agp_b}`\n",
             "Flag = |Δ| exceeds sqrt(ci_A² + ci_B²) (outside the combined 95% CI). "
             "Random-init + shuffled K-fold ⇒ some drift is expected.\n"]
    for label in LABELS:
        lines += [f"\n## `{label}`  (`{CFG[label]}`)\n",
                  "| Task | A mean | B mean | Δ | combined CI | flag |",
                  "|---|---|---|---|---|---|"]
        A = scratch_col(label, hmc_a, agp_a)
        B = scratch_col(label, hmc_b, agp_b)
        for t in HMC_TASKS + AGP_TASKS:
            pa, pb = A.get(t, {}).get("AUROC"), B.get(t, {}).get("AUROC")
            if not pa or pa[0] is None or not pb or pb[0] is None:
                lines.append(f"| {t} | {cell(pa)} | {cell(pb)} | – | – | (missing) |")
                continue
            d = pb[0] - pa[0]
            comb = math.sqrt((pa[1] or 0.0) ** 2 + (pb[1] or 0.0) ** 2)
            flag = "⚠️" if comb and abs(d) > comb else ""
            lines.append(f"| {t} | {pa[0]:.4f} | {pb[0]:.4f} | {d:+.4f} | "
                         f"{comb:.4f} | {flag} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch-hmc-root", default=DEF_SCRATCH_HMC)
    ap.add_argument("--scratch-agp-root", default=DEF_SCRATCH_AGP)
    ap.add_argument("--out", default=os.path.join(V2, "scratch_vs_pretrained.md"))
    ap.add_argument("--reconcile", nargs=2, metavar=("HMC_ROOT", "AGP_ROOT"),
                    help="compare the default scratch run against a re-run under these roots")
    args = ap.parse_args()

    if args.reconcile:
        txt = reconcile(args.scratch_hmc_root, args.scratch_agp_root,
                        args.reconcile[0], args.reconcile[1])
        out = args.out.replace(".md", "") + "_reconcile.md"
        with open(out, "w") as f:
            f.write(txt)
        print(txt)
        print(f"\nwrote {out}")
        return

    data = load_all(args.scratch_hmc_root, args.scratch_agp_root)
    parts = []
    for label in LABELS:
        parts.append(build(label, data[label]))
    txt = "\n\n---\n\n".join(parts) + "\n"
    with open(args.out, "w") as f:
        f.write(txt)
    print(txt)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
