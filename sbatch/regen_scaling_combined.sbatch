#!/bin/bash
#SBATCH --job-name=regen_scaling_combined
#SBATCH -A aip-brudno
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=0:15:00
#SBATCH --output=logs/refactored/pretrain/scaling/regen_%j.out
#SBATCH --error=logs/refactored/pretrain/scaling/regen_%j.err

# Regenerate the two combined data-scaling result files. Submit with
#   --dependency=afterany:<finetune_scaling_refs job id>
# so the 100%-ref column + error bars refresh once the ref eval lands.
set -euo pipefail
source ~/hmbenv/bin/activate
cd /project/aip-brudno/kchen13/gut_microbiome_GPT
python scripts/combine_scaling_summaries.py
