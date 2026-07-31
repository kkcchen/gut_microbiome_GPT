#!/bin/bash
#SBATCH --job-name=mgm_downstream_final
#SBATCH -A aip-brudno
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --output=logs/downstream_final/mgm/%A_%a.out
#SBATCH --error=logs/downstream_final/mgm/%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --array=0-10

# One task per array job. Each job writes only its own directory under
# outputs/downstream_final/<task>/, so nothing races; the run-wide tables in
# results/ are built afterwards by the --collect step below.
#
# Before first submission:
#   mkdir -p logs/downstream_final
#
# After the array finishes (takes seconds, fine on a login node):
#   python kd_tasks/run_downstream_final.py --collect
#

# prepare your environment here
source ~/hmbenv/bin/activate
cd /project/aip-brudno/kchen13/gut_microbiome_GPT

# Order and length must match --array above (11 tasks, 0-10).
# Quoted throughout so task names containing spaces survive.
TASKS=(
    "location"
    "sex"
    "supplement"
    "helicobacter_pylori_infection"
    "hbv_infection"
    "sunflower_seed_oil_emollient_therapy"
    "crohns_disease"
    "diarrhea"
    "type_1_diabetes"
    "hiv_infection"
    "chemotherapy"
)

TASK="${TASKS[$SLURM_ARRAY_TASK_ID]}"
echo "array index $SLURM_ARRAY_TASK_ID -> task '$TASK'"

python kd_tasks/run_downstream_final.py --tasks "$TASK"
