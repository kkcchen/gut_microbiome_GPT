#!/bin/bash
#SBATCH --job-name=regen_scaling_combined
#SBATCH -A aip-brudno
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=0:15:00
#SBATCH --output=logs/refactored/pretrain/scaling/regen_%j.out
#SBATCH --error=logs/refactored/pretrain/scaling/regen_%j.err

# Regenerate the combined data-scaling result files (v1 + v2). Submit with
#   --dependency=afterany:<finetune job id>
# so the tables refresh once the eval lands.
set -euo pipefail
source ~/hmbenv/bin/activate
cd /project/aip-brudno/kchen13/gut_microbiome_GPT
python scripts/combine_scaling_summaries.py
python scripts/combine_study_scaling.py
