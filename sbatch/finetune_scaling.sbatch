#!/bin/bash
#SBATCH --job-name=finetune_scaling
#SBATCH -A aip-brudno
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --array=0-19%8
#SBATCH --gres=gpu:1
#SBATCH --output=logs/refactored/pretrain/scaling/ft_%a_%j.out
#SBATCH --error=logs/refactored/pretrain/scaling/ft_%a_%j.err

# Downstream evaluation for the data-scaling experiment (v1). One array task per
# (checkpoint, task-set); NO re-pretraining -- scripts.finetune_all_tasks reads each
# existing checkpoint from its pretrain config's paths.output_dir.
#
# 10 checkpoints x {HMC 13 tasks, AGP 18 tasks} = 20 array tasks:
#   ckpt = idx // 2 ; is_agp = idx % 2
#   0-3  baseline scaling r010/r020/r050/r100
#   4-7  stage3   scaling r010/r020/r050/r100
#   8    baseline 100%-ref (original July checkpoint)
#   9    stage3   100%-ref
#
# FT_MODE env var selects the finetuning regime (default: partial):
#   FT_MODE=partial  frozen transformer + MLP head (linear-probe readout)
#   FT_MODE=full     unfreeze the whole transformer, train end-to-end
#
# Outputs:  outputs/pretrain/scaling/<FT_MODE>/<name>/       (HMC)
#           outputs/pretrain/scaling/<FT_MODE>_agp/<name>/    (AGP)
# read by scripts/combine_scaling_summaries.py.
#
# Usage:  sbatch sbatch/finetune_scaling.sbatch                       # partial
#         FT_MODE=full sbatch --export=ALL,FT_MODE=full sbatch/finetune_scaling.sbatch

set -euo pipefail

source ~/hmbenv/bin/activate
cd /project/aip-brudno/kchen13/gut_microbiome_GPT

FT_MODE="${FT_MODE:-partial}"

# "<pretrain-config> <output-name>", one per checkpoint
CKPTS=(
    "configs/pretrain/scaling/scaling_baseline_bins20_mask030_r010.yaml scaling_baseline_bins20_mask030_r010"
    "configs/pretrain/scaling/scaling_baseline_bins20_mask030_r020.yaml scaling_baseline_bins20_mask030_r020"
    "configs/pretrain/scaling/scaling_baseline_bins20_mask030_r050.yaml scaling_baseline_bins20_mask030_r050"
    "configs/pretrain/scaling/scaling_baseline_bins20_mask030_r100.yaml scaling_baseline_bins20_mask030_r100"
    "configs/pretrain/scaling/scaling_stage3_concatenation_r010.yaml scaling_stage3_concatenation_r010"
    "configs/pretrain/scaling/scaling_stage3_concatenation_r020.yaml scaling_stage3_concatenation_r020"
    "configs/pretrain/scaling/scaling_stage3_concatenation_r050.yaml scaling_stage3_concatenation_r050"
    "configs/pretrain/scaling/scaling_stage3_concatenation_r100.yaml scaling_stage3_concatenation_r100"
    "configs/pretrain/pretrained_baseline/pretrained_baseline_bins20_mask030.yaml baseline_bins20_mask030"
    "configs/pretrain/real_runs/stage3_abundance_emb_concatenation.yaml stage3_concatenation"
)

HMC_TASKS="age,bmi,sex,supplement,helicobacter_pylori_infection,hbv_infection,hiv_infection,sunflower_seed_oil_emollient_therapy,chemotherapy,type_1_diabetes,crohns_disease,diarrhea,location"
AGP_TASKS="age cat,alcohol_frequency,antibiotic_history,asd,autoimmune,contraceptive,diabetes,fruit_frequency,gluten,ibd,ibs,meat_eggs_frequency,probiotic_frequency,sibo,sleep_duration,sugary_sweets_frequency,vegetable_frequency,weight_change"

ckpt_idx=$(( SLURM_ARRAY_TASK_ID / 2 ))
is_agp=$(( SLURM_ARRAY_TASK_ID % 2 ))
read -r CONFIG NAME <<< "${CKPTS[$ckpt_idx]}"

echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID  ft_mode=$FT_MODE  config=$CONFIG  name=$NAME  agp=$is_agp"

if [[ "$is_agp" -eq 0 ]]; then
    python -m scripts.finetune_all_tasks \
        --config "${CONFIG}" \
        --finetune-output-root "outputs/pretrain/scaling/${FT_MODE}/${NAME}" \
        --tasks "${HMC_TASKS}" \
        finetune.training.finetune_mode="${FT_MODE}"
else
    python -m scripts.finetune_all_tasks \
        --config "${CONFIG}" \
        --finetune-output-root "outputs/pretrain/scaling/${FT_MODE}_agp/${NAME}" \
        --tasks "${AGP_TASKS}" \
        finetune.paths.downstream_train=/scratch/kchen13/hmc_final_fixed/downstream_agp_train.h5ad \
        finetune.paths.downstream_test=/scratch/kchen13/hmc_final_fixed/downstream_agp_test.h5ad \
        finetune.training.finetune_mode="${FT_MODE}"
fi
