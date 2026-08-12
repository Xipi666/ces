#!/usr/bin/env bash

set -o pipefail

usage() {
    cat <<'EOF'
Usage: run_carm_lr_sweep.sh [options]

Options:
  --python PATH       Python interpreter (default: python3)
  --data-dir PATH     Dataset directory (default: SCRIPT_DIR/dataset)
  --log-dir PATH      Log directory (default: SCRIPT_DIR/logs/carm_lr_sweep)
  --gpu-id ID         CUDA device id (default: 0)
  --seq-len N         Input sequence length (default: 96)
  --epochs N          Training epochs (default: 100)
  --oof-epochs N      OOF epochs (default: 100)
  --oof-folds N       OOF folds (default: 5)
  --gate-epochs N     Gate epoch-selection limit (default: 80)
  --gate-patience N   Gate early-stopping patience (default: 15)
  --gate-min-epochs N Gate minimum epochs before early stop (default: 15)
  --batch-size N      Batch size (default: 32)
  --patience N        Early-stopping patience (default: 15)
  --min-epochs N      Minimum epochs (default: 15)
  --top-k N           Number of selected experts (default: 8)
  --key-points N      Number of key points (default: 32)
  --seed N            Random seed (default: 2024)
  --alpha-sparsity-weight X
                       L1 penalty for horizon-wise alpha (default: 0.01)
  --alpha-init X      Initial alpha value in (0, 1) (default: 0.02)
  -h, --help          Show this help
EOF
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_path="python3"
data_dir="$script_dir/dataset"
log_dir="$script_dir/logs/carm_lr_sweep"
gpu_id=0
seq_len=96
epochs=100
oof_epochs=100
oof_folds=5
gate_epochs=80
gate_patience=15
gate_min_epochs=15
batch_size=32
patience=15
min_epochs=15
top_k=8
key_points=32
seed=2024
alpha_sparsity_weight=0.01
alpha_init=0.02

while [[ $# -gt 0 ]]; do
    case "$1" in
        --python)       python_path="$2"; shift 2 ;;
        --data-dir)     data_dir="$2"; shift 2 ;;
        --log-dir)      log_dir="$2"; shift 2 ;;
        --gpu-id)       gpu_id="$2"; shift 2 ;;
        --seq-len)      seq_len="$2"; shift 2 ;;
        --epochs)       epochs="$2"; shift 2 ;;
        --oof-epochs)   oof_epochs="$2"; shift 2 ;;
        --oof-folds)    oof_folds="$2"; shift 2 ;;
        --gate-epochs)  gate_epochs="$2"; shift 2 ;;
        --gate-patience) gate_patience="$2"; shift 2 ;;
        --gate-min-epochs) gate_min_epochs="$2"; shift 2 ;;
        --batch-size)   batch_size="$2"; shift 2 ;;
        --patience)     patience="$2"; shift 2 ;;
        --min-epochs)   min_epochs="$2"; shift 2 ;;
        --top-k)        top_k="$2"; shift 2 ;;
        --key-points)   key_points="$2"; shift 2 ;;
        --seed)         seed="$2"; shift 2 ;;
        --alpha-sparsity-weight) alpha_sparsity_weight="$2"; shift 2 ;;
        --alpha-init)   alpha_init="$2"; shift 2 ;;
        -h|--help)      usage; exit 0 ;;
        *)              echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ "$python_path" == */* ]]; then
    [[ -f "$python_path" ]] || { echo "Python interpreter not found: $python_path" >&2; exit 1; }
else
    command -v "$python_path" >/dev/null 2>&1 || { echo "Python interpreter not found: $python_path" >&2; exit 1; }
fi

main_script="$script_dir/ces_hmoe_carm_ettdataset.py"
if [[ ! -f "$main_script" ]]; then
    echo "CARM script not found: $main_script" >&2
    exit 1
fi

mkdir -p "$log_dir"

datasets=(ETTm1 ETTh1)
pred_lens=(24 96 192 336 720)
learning_rates=(0.00001)
gate_learning_rates=(0.0001)
total=$(( ${#datasets[@]} * ${#pred_lens[@]} * ${#learning_rates[@]} * ${#gate_learning_rates[@]} ))
count=0
fail_count=0

echo "CARM learning-rate sweep: $total experiments"
echo "DataDir=$data_dir LogDir=$log_dir SeqLen=$seq_len Epochs=$epochs OofEpochs=$oof_epochs OofFolds=$oof_folds GateEpochs=$gate_epochs GatePatience=$gate_patience GateMinEpochs=$gate_min_epochs"
echo "Datasets=${datasets[*]} PredLens=${pred_lens[*]}"
echo "LearningRates=${learning_rates[*]} GateLearningRates=${gate_learning_rates[*]}"
echo "AlphaSparsityWeight=$alpha_sparsity_weight AlphaInit=$alpha_init"

rate_tag() {
    local value="$1"
    awk -v value="$value" 'BEGIN {
        result = sprintf("%.5f", value)
        sub(/0+$/, "", result)
        sub(/\.$/, "", result)
        gsub(/\./, "p", result)
        print result
    }'
}

for dataset in "${datasets[@]}"; do
    for pred_len in "${pred_lens[@]}"; do
        for learning_rate in "${learning_rates[@]}"; do
            for gate_learning_rate in "${gate_learning_rates[@]}"; do
                count=$((count + 1))
                lr_tag="$(rate_tag "$learning_rate")"
                gate_lr_tag="$(rate_tag "$gate_learning_rate")"
                log_file="$log_dir/${dataset}_pl${pred_len}_lr${lr_tag}_gate${gate_lr_tag}.log"

                echo "[$count/$total] $dataset | seq_len=$seq_len | pred_len=$pred_len | lr=$learning_rate | gate_lr=$gate_learning_rate"
                CUDA_VISIBLE_DEVICES="$gpu_id" "$python_path" -u "$main_script" \
                    --dataset "$dataset" \
                    --pred_len "$pred_len" \
                    --seq_len "$seq_len" \
                    --data_dir "$data_dir" \
                    --epochs "$epochs" \
                    --oof_epochs "$oof_epochs" \
                    --oof_folds "$oof_folds" \
                    --gate_epochs "$gate_epochs" \
                    --gate_patience "$gate_patience" \
                    --gate_min_epochs "$gate_min_epochs" \
                    --batch_size "$batch_size" \
                    --lr "$learning_rate" \
                    --gate_lr "$gate_learning_rate" \
                    --alpha_sparsity_weight "$alpha_sparsity_weight" \
                    --alpha_init "$alpha_init" \
                    --patience "$patience" \
                    --min_epochs "$min_epochs" \
                    --top_k "$top_k" \
                    --key_points "$key_points" \
                    --seed "$((seed + count - 1))" \
                    --device auto \
                    --balance_weight 0.01 2>&1 | tee "$log_file"
                exit_code=${PIPESTATUS[0]}

                if [[ $exit_code -eq 0 ]]; then
                    echo "  completed | log=$log_file"
                else
                    fail_count=$((fail_count + 1))
                    echo "  FAILED (exit=$exit_code) | log=$log_file"
                fi
            done
        done
    done
done

success=$((total - fail_count))
echo "Finished: $success/$total succeeded; $fail_count failed"
[[ $fail_count -eq 0 ]]
