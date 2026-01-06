#!/bin/bash
#SBATCH --job-name=test
#SBATCH -A aip-rahulgk
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=0:10:00
#SBATCH --output=logs/test/%j.out
#SBATCH --error=logs/test/%j.err

# --- optional but recommended ---
module load python/3.11  # or whatever your cluster uses
source ~/hmbenv/bin/activate

# --- run the script ---
python /home/kchen13/projects/aip-rahulgk/kchen13/gutmodel/gut_microbiome_GPT/notebooks/share_taxonomy.py
