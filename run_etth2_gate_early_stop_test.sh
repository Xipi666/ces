#!/usr/bin/env bash

set -o pipefail

usage() {
    cat <<'EOF'
Usage: run_etth2_gate_early_stop_test.sh [options]

Compare gate OOF5 early stopping with a full gate epoch search on ETTh2.
Both modes use the same seed for each prediction length.

Options:
  --python PATH          Python interpreter (default: python3)
  --data-dir PATH        Dataset directory (default: SCRIPT_DIR/dataset)
  --log-dir PATH         Log directory (default: SCRIPT_DIR/logs/etth2_gate_early_stop)
  --gpu-id ID            CUDA device id (default: 0)
  --epochs N             Final/base-model max epochs (default: 100)
  --oof-epochs N         OOF base-model max epochs (default: 100)
  --gate-epochs N        Gate epoch-search limit (default: 80)
  --gate-patience N      Gate early-stopping patience (default: 15)
  --gate-min-epochs N    Minimum epochs before early stop (default: 15)
  --batch-size N         Batch size (default: 32)
  --seed N               Base random seed (default: 2024)
  -h, --help             Show this help
EOF
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_path="python3"
data_dir="$script_dir/dataset"
log_dir="$script_dir/logs/etth2_gate_early_stop"
gpu_id=0
epochs=100
oof_epochs=100
gate_epochs=80
gate_patience=15
gate_min_epochs=15
batch_size=32
seed=2024

while [[ $# -gt 0 ]]; do
    case "$1" in
        --python)          python_path="$2"; shift 2 ;;
        --data-dir)        data_dir="$2"; shift 2 ;;
        --log-dir)         log_dir="$2"; shift 2 ;;
        --gpu-id)          gpu_id="$2"; shift 2 ;;
        --epochs)          epochs="$2"; shift 2 ;;
        --oof-epochs)      oof_epochs="$2"; shift 2 ;;
        --gate-epochs)     gate_epochs="$2"; shift 2 ;;
        --gate-patience)   gate_patience="$2"; shift 2 ;;
        --gate-min-epochs) gate_min_epochs="$2"; shift 2 ;;
        --batch-size)      batch_size="$2"; shift 2 ;;
        --seed)            seed="$2"; shift 2 ;;
        -h|--help)         usage; exit 0 ;;
        *)                 echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
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
if [[ "$gate_min_epochs" -gt "$gate_epochs" ]]; then
    echo "--gate-min-epochs cannot be greater than --gate-epochs" >&2
    exit 2
fi

mkdir -p "$log_dir"
summary_file="$log_dir/summary.tsv"
printf 'mode\tpred_len\tseed\tbest_epoch\tstop_epoch\tbest_val_mse\tlast_val_mse\tdiagnosis\ttest_mse\ttest_base_mse\tstatus\n' > "$summary_file"

pred_lens=(24 96 192 336 720)
modes=(early_stop full_search)
total=$(( ${#pred_lens[@]} * ${#modes[@]} ))
count=0
fail_count=0

echo "ETTh2 gate early-stopping test: $total experiments"
echo "dataset=ETTh2 seq_len=96 lr=0.00001 gate_lr=0.0001 pred_lens=${pred_lens[*]}"
echo "early_stop: gate_epochs=$gate_epochs gate_patience=$gate_patience gate_min_epochs=$gate_min_epochs"
echo "full_search: gate_epochs=$gate_epochs gate_patience=$gate_epochs gate_min_epochs=$gate_epochs"
echo "summary=$summary_file"

extract_last() {
    local pattern="$1"
    local file="$2"
    sed -nE "$pattern" "$file" | tail -n 1
}

append_summary() {
    local mode="$1"
    local pred_len="$2"
    local run_seed="$3"
    local log_file="$4"
    local status="$5"
    local best_epoch stop_epoch best_val_mse last_val_mse diagnosis test_mse test_base_mse

    best_epoch="$(extract_last 's/.*Gate selection result: best_epoch=([0-9]+).*/\1/p' "$log_file")"
    stop_epoch="$(extract_last 's/.*Gate select early stopping at epoch ([0-9]+).*/\1/p' "$log_file")"
    best_val_mse="$(extract_last 's/.*Gate selection result: best_epoch=[0-9]+ best_val_mse=([^ ]+).*/\1/p' "$log_file")"
    last_val_mse="$(extract_last 's/.*last_val_mse=([^ ]+) diagnosis=.*/\1/p' "$log_file")"
    diagnosis="$(extract_last 's/.*diagnosis=([^ ]+).*/\1/p' "$log_file")"
    test_mse="$(extract_last "s/.*Selected Test:.*'mse': ([0-9.eE+-]+).*/\\1/p" "$log_file")"
    test_base_mse="$(extract_last "s/.*Selected Test:.*'base_mse': ([0-9.eE+-]+).*/\\1/p" "$log_file")"

    best_epoch=${best_epoch:--}
    stop_epoch=${stop_epoch:--}
    best_val_mse=${best_val_mse:--}
    last_val_mse=${last_val_mse:--}
    diagnosis=${diagnosis:--}
    test_mse=${test_mse:--}
    test_base_mse=${test_base_mse:--}
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$mode" "$pred_len" "$run_seed" "$best_epoch" "$stop_epoch" \
        "$best_val_mse" "$last_val_mse" "$diagnosis" "$test_mse" "$test_base_mse" "$status" \
        >> "$summary_file"
}

for index in "${!pred_lens[@]}"; do
    pred_len="${pred_lens[$index]}"
    run_seed=$((seed + index))
    for mode in "${modes[@]}"; do
        count=$((count + 1))
        if [[ "$mode" == "early_stop" ]]; then
            run_patience="$gate_patience"
            run_min_epochs="$gate_min_epochs"
        else
            # Run all gate epochs while keeping OOF5 best-state selection.
            run_patience="$gate_epochs"
            run_min_epochs="$gate_epochs"
        fi

        log_file="$log_dir/ETTh2_pl${pred_len}_${mode}.log"
        echo "[$count/$total] ETTh2 | seq_len=96 | pred_len=$pred_len | mode=$mode | seed=$run_seed"
        CUDA_VISIBLE_DEVICES="$gpu_id" "$python_path" -u "$main_script" \
            --dataset ETTh2 \
            --data_dir "$data_dir" \
            --seq_len 96 \
            --pred_len "$pred_len" \
            --epochs "$epochs" \
            --oof_epochs "$oof_epochs" \
            --oof_folds 5 \
            --gate_epochs "$gate_epochs" \
            --gate_patience "$run_patience" \
            --gate_min_epochs "$run_min_epochs" \
            --batch_size "$batch_size" \
            --lr 1e-5 \
            --gate_lr 1e-4 \
            --patience 15 \
            --min_epochs 15 \
            --top_k 8 \
            --key_points 32 \
            --seed "$run_seed" \
            --device auto \
            --balance_weight 0.01 2>&1 | tee "$log_file"
        exit_code=${PIPESTATUS[0]}

        if [[ $exit_code -eq 0 ]]; then
            append_summary "$mode" "$pred_len" "$run_seed" "$log_file" "ok"
            echo "  completed | log=$log_file"
        else
            fail_count=$((fail_count + 1))
            append_summary "$mode" "$pred_len" "$run_seed" "$log_file" "failed:$exit_code"
            echo "  FAILED (exit=$exit_code) | log=$log_file"
        fi
    done
done

echo
echo "Summary:"
if command -v column >/dev/null 2>&1; then
    column -t -s $'\t' "$summary_file"
else
    cat "$summary_file"
fi
echo
echo "Compare early_stop and full_search for each pred_len in $summary_file."
echo "Early stopping is useful when it stops before gate_epochs without worsening test_mse."
echo "Finished: $((total - fail_count))/$total succeeded; $fail_count failed"
[[ $fail_count -eq 0 ]]
