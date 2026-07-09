#!/bin/bash
#SBATCH --job-name=finetune_real_runs_categorical
#SBATCH -A aip-brudno
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/finetune/real_runs/%x_%j.out
#SBATCH --error=logs/finetune/real_runs/%x_%j.err

# Finetunes one pretrained checkpoint from configs/pretrain/real_runs/<RUN_NAME>.yaml
# on every categorical (classification) downstream task, skipping the continuous
# (regression) tasks age/bmi.
#
# Usage: sbatch sbatch/finetune_real_runs_categorical.sbatch <run_name>
#   e.g.  sbatch sbatch/finetune_real_runs_categorical.sbatch bins10_mask015

set -uo pipefail

RUN_NAME=${1:?"Usage: sbatch finetune_real_runs_categorical.sbatch <run_name, e.g. bins10_mask015>"}

source ~/hmbenv/bin/activate
cd /project/aip-brudno/kchen13/gut_microbiome_GPT

python -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available. Exiting.'" || exit 1

PRETRAIN_CONFIG="configs/pretrain/real_runs/${RUN_NAME}.yaml"
if [ ! -f "$PRETRAIN_CONFIG" ]; then
    echo "ERROR: no such pretrain config: ${PRETRAIN_CONFIG}"
    exit 1
fi

PRETRAIN_OUTPUT_DIR="outputs/pretrain/real_runs/${RUN_NAME}"
CHECKPOINT_PATH="${PRETRAIN_OUTPUT_DIR}/best_model/best_model/pytorch_model.bin"
TAXA_VOCAB_PATH="${PRETRAIN_OUTPUT_DIR}/taxa_vocab.pkl"

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "ERROR: checkpoint not found at ${CHECKPOINT_PATH} -- has the ${RUN_NAME} pretrain job finished?"
    exit 1
fi
if [ ! -f "$TAXA_VOCAB_PATH" ]; then
    echo "ERROR: taxa vocab not found at ${TAXA_VOCAB_PATH}"
    exit 1
fi

# Derive architecture-matching overrides from the chosen pretrain config, rather than
# assuming configs/finetune/default.yaml's model.params already match -- they don't
# (default.yaml is a 7-layer/log_rel_abundance template; the real_runs sweep uses
# 3 layers/binning). Mismatched model.params would make the strict=False checkpoint
# load silently skip transformer layers instead of loading pretrained weights.
eval "$(python3 - "$PRETRAIN_CONFIG" <<'PYEOF'
import sys
from omegaconf import OmegaConf

cfg = OmegaConf.load(sys.argv[1])
p = cfg.model.params

def b(x):
    return str(x).lower()

print(f"D_MODEL={p.d_model}")
print(f"NHEAD={p.nhead}")
print(f"D_HID={p.d_hid}")
print(f"NLAYERS={p.nlayers}")
print(f"DROPOUT={p.dropout}")
print(f"EMB_STYLE={p.abundance_emb_style}")
print(f"SAMPLE_EMB_STYLE={p.sample_emb_style}")
print(f"USE_GNN={b(p.use_gnn)}")
print(f"NORM_STRATEGY={cfg.data.norm_strategy}")
print(f"NUM_BINS={cfg.data.num_bins}")
PYEOF
)"

echo "Finetuning from pretrain run: ${RUN_NAME}"
echo "  Checkpoint:  ${CHECKPOINT_PATH}"
echo "  Taxa vocab:  ${TAXA_VOCAB_PATH}"
echo "  Architecture: d_model=${D_MODEL} nhead=${NHEAD} d_hid=${D_HID} nlayers=${NLAYERS} dropout=${DROPOUT} abundance_emb_style=${EMB_STYLE} sample_emb_style=${SAMPLE_EMB_STYLE} use_gnn=${USE_GNN}"
echo "  Data: norm_strategy=${NORM_STRATEGY} num_bins=${NUM_BINS}"

TRAIN_H5AD="/scratch/kchen13/hmc_final_fixed/downstream_train.h5ad"
TEST_H5AD="/scratch/kchen13/hmc_final_fixed/downstream_test.h5ad"

BASE_CONFIG="configs/finetune/default.yaml"
OUTPUT_ROOT="outputs/finetune/real_runs/${RUN_NAME}"
LOG_DIR="logs/finetune/real_runs/${RUN_NAME}"
mkdir -p "$LOG_DIR"

# Categorical (classification) downstream tasks only. age/bmi are continuous
# (regression) and are deliberately excluded -- see configs/eval/default.yaml's
# downstream_tasks_config.tasks for the canonical task registry.
TASKS=(
    location
    sex
    supplement
    helicobacter_pylori_infection
    hbv_infection
    hiv_infection
    sunflower_seed_oil_emollient_therapy
    chemotherapy
    type_1_diabetes
    crohns_disease
    diarrhea
)

FAILED_TASKS=()
SUCCEEDED_TASKS=()

for TASK in "${TASKS[@]}"; do
    echo "=================================================================="
    echo "Finetuning task: ${TASK}"
    echo "=================================================================="

    TASK_LOG="${LOG_DIR}/${TASK}.log"

    
        accelerate launch -m scripts.finetune --config "${BASE_CONFIG}" \
        paths.downstream_train="${TRAIN_H5AD}" \
        paths.downstream_test="${TEST_H5AD}" \
        paths.output_dir="${OUTPUT_ROOT}/${TASK}" \
        paths.checkpoint_path="${CHECKPOINT_PATH}" \
        paths.taxa_vocab_path="${TAXA_VOCAB_PATH}" \
        data.finetune_task_name="${TASK}" \
        data.label_column=categorical_label \
        data.norm_strategy="${NORM_STRATEGY}" \
        data.num_bins="${NUM_BINS}" \
        training.finetune_task=classification \
        model.params.d_model="${D_MODEL}" \
        model.params.nhead="${NHEAD}" \
        model.params.d_hid="${D_HID}" \
        model.params.nlayers="${NLAYERS}" \
        model.params.dropout="${DROPOUT}" \
        model.params.abundance_emb_style="${EMB_STYLE}" \
        model.params.sample_emb_style="${SAMPLE_EMB_STYLE}" \
        model.params.use_gnn="${USE_GNN}" \
        wandb.run_name="${RUN_NAME}_finetune_${TASK}" \
        > "${TASK_LOG}" 2>&1

    if [ $? -eq 0 ]; then
        echo "SUCCESS: ${TASK} (log: ${TASK_LOG})"
        SUCCEEDED_TASKS+=("$TASK")
    else
        echo "FAILED: ${TASK} (see log: ${TASK_LOG})"
        FAILED_TASKS+=("$TASK")
    fi
done

echo "=================================================================="
echo "Finetuning sweep complete for pretrain run: ${RUN_NAME}"
echo "Succeeded (${#SUCCEEDED_TASKS[@]}/${#TASKS[@]}): ${SUCCEEDED_TASKS[*]:-none}"
echo "Failed (${#FAILED_TASKS[@]}/${#TASKS[@]}): ${FAILED_TASKS[*]:-none}"
echo "=================================================================="

if [ ${#FAILED_TASKS[@]} -gt 0 ]; then
    exit 1
fi
