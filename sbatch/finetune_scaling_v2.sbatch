#!/bin/bash
#SBATCH --job-name=finetune_scaling_v2
#SBATCH -A aip-brudno
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --array=0-15%8
#SBATCH --gres=gpu:1
#SBATCH --output=logs/refactored/pretrain/scaling_v2/full_%a_%j.out
#SBATCH --error=logs/refactored/pretrain/scaling_v2/full_%a_%j.err

# Data-scaling v2 downstream eval: FULL finetuning (whole transformer unfrozen,
# finetune.training.finetune_mode=full) of the 8 v2 pretrain checkpoints, pooled
# 5-fold CV, HMC + AGP. NO re-pretraining -- scripts.finetune_all_tasks reads each
# checkpoint from its config's paths.output_dir (outputs/pretrain/scaling_v2/...).
#
# 8 checkpoints x {HMC 13 tasks, AGP 18 tasks} = 16 array tasks:
#   ckpt = idx // 2 ; is_agp = idx % 2
#   0-3  baseline r010/r020/r050/r100 ; 4-7  stage3 r010/r020/r050/r100
#
# Outputs: outputs/pretrain/scaling_v2/full/<name>/  and  full_agp/<name>/
# read by scripts/combine_study_scaling.py.

set -euo pipefail

source ~/hmbenv/bin/activate
cd /project/aip-brudno/kchen13/gut_microbiome_GPT
mkdir -p logs/refactored/pretrain/scaling_v2

# "<pretrain-config> <output-name>", one per checkpoint (name == checkpoint dir name)
CKPTS=(
    "configs/pretrain/scaling/scaling_v2_baseline_bins20_mask030_r010.yaml scaling_baseline_bins20_mask030_r010"
    "configs/pretrain/scaling/scaling_v2_baseline_bins20_mask030_r020.yaml scaling_baseline_bins20_mask030_r020"
    "configs/pretrain/scaling/scaling_v2_baseline_bins20_mask030_r050.yaml scaling_baseline_bins20_mask030_r050"
    "configs/pretrain/scaling/scaling_v2_baseline_bins20_mask030_r100.yaml scaling_baseline_bins20_mask030_r100"
    "configs/pretrain/scaling/scaling_v2_stage3_concatenation_r010.yaml scaling_stage3_concatenation_r010"
    "configs/pretrain/scaling/scaling_v2_stage3_concatenation_r020.yaml scaling_stage3_concatenation_r020"
    "configs/pretrain/scaling/scaling_v2_stage3_concatenation_r050.yaml scaling_stage3_concatenation_r050"
    "configs/pretrain/scaling/scaling_v2_stage3_concatenation_r100.yaml scaling_stage3_concatenation_r100"
)

HMC_TASKS="age,bmi,sex,supplement,helicobacter_pylori_infection,hbv_infection,hiv_infection,sunflower_seed_oil_emollient_therapy,chemotherapy,type_1_diabetes,crohns_disease,diarrhea,location"
AGP_TASKS="age cat,alcohol_frequency,antibiotic_history,asd,autoimmune,contraceptive,diabetes,fruit_frequency,gluten,ibd,ibs,meat_eggs_frequency,probiotic_frequency,sibo,sleep_duration,sugary_sweets_frequency,vegetable_frequency,weight_change"

ckpt_idx=$(( SLURM_ARRAY_TASK_ID / 2 ))
is_agp=$(( SLURM_ARRAY_TASK_ID % 2 ))
read -r CONFIG NAME <<< "${CKPTS[$ckpt_idx]}"

echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID  config=$CONFIG  name=$NAME  agp=$is_agp"

if [[ "$is_agp" -eq 0 ]]; then
    python -m scripts.finetune_all_tasks \
        --config "${CONFIG}" \
        --finetune-output-root "outputs/pretrain/scaling_v2/full/${NAME}" \
        --tasks "${HMC_TASKS}" \
        finetune.training.finetune_mode=full
else
    python -m scripts.finetune_all_tasks \
        --config "${CONFIG}" \
        --finetune-output-root "outputs/pretrain/scaling_v2/full_agp/${NAME}" \
        --tasks "${AGP_TASKS}" \
        finetune.paths.downstream_train=/scratch/kchen13/hmc_final_fixed/downstream_agp_train.h5ad \
        finetune.paths.downstream_test=/scratch/kchen13/hmc_final_fixed/downstream_agp_test.h5ad \
        finetune.training.finetune_mode=full
fi
