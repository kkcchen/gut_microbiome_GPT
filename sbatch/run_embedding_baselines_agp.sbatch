#!/bin/bash
#SBATCH --job-name=embedding_baselines_agp
#SBATCH -A aip-brudno
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --output=logs/with_seeds/embedding_baselines_agp/%a_%j.out
#SBATCH --error=logs/with_seeds/embedding_baselines_agp/%a_%j.err

# One SLURM array task per winning checkpoint, running
# scripts/run_embedding_baselines_with_seeds.sh (unmodified) restricted to one of the two
# new AGP configs (via EMBED_CONFIGS) against downstream_agp_{train,test}.h5ad -- see
# scripts/split_agp_downstream_tasks.py for how those were produced from the single
# pre-aggregated downstream_agp.h5ad. Every task in that data is single-study, so it all
# runs through the existing pooled 5-fold CV path (utils/downstream_utils.py::
# run_task_combined_cv) -- no presplit tasks here, no seed loop needed.
#
# 24h (vs 12h for the HMC with_seeds run): 18 tasks with larger per-task sample counts
# (~4,200-4,541 rows each) than any single-study HMC task, so substantially more XGBoost
# grid-search fold-fits.
#
# Usage: sbatch sbatch/run_embedding_baselines_agp.sbatch

set -euo pipefail

source ~/hmbenv/bin/activate
cd /project/aip-brudno/kchen13/gut_microbiome_GPT

AGP_CONFIGS_LIST=(
    "configs/eval/embedding_baselines/pretrained_baseline_bins20_mask030_agp.yaml"
    "configs/eval/embedding_baselines/stage3_abundance_emb_concatenation_agp.yaml"
)
CONFIG="${AGP_CONFIGS_LIST[$SLURM_ARRAY_TASK_ID]}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID} SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "Using config: ${CONFIG}"

EMBED_CONFIGS="${CONFIG}" scripts/run_embedding_baselines_with_seeds.sh "outputs/eval/embedding_baselines_agp"
