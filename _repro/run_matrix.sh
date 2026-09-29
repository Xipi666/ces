#!/usr/bin/env bash
# Reproduce the Aug 20-26 Codex "best model" experiment matrix
# (run_best_model_4datasets.ps1, MasterSeed=2024, TrendExpert-v2/PeriodicExpert-v2/RampExpert-v1)
# Usage: run_matrix.sh DATASET MASTERSEED [SEQLENS] [PREDLENS]
set -o pipefail

PYTHON="/g/Anaconda3/envs/wrj/python.exe"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MAIN="$SCRIPT_DIR/ces_hmoe_ettdataset.py"
DATA_DIR="$SCRIPT_DIR/dataset"

DATASET="${1:-ETTh2}"
SEED="${2:-2024}"
SEQLENS="${3:-96 168 336 720}"
PREDLENS="${4:-24 96 192 336 720}"

LOG_ROOT="$SCRIPT_DIR/_repro/logs/${DATASET}_matrix_seed${SEED}"
mkdir -p "$LOG_ROOT"

# Prevent concurrent runner instances (lock is stolen if the holder died)
LOCK="$SCRIPT_DIR/_repro/.matrix_lock"
if mkdir "$LOCK" 2>/dev/null; then
    echo $$ > "$LOCK/pid"
    trap 'rm -rf "$LOCK"' EXIT
else
    holder=$(cat "$LOCK/pid" 2>/dev/null || echo "?")
    if [[ "$holder" != "?" ]] && ! kill -0 "$holder" 2>/dev/null; then
        echo "stale lock from dead pid $holder; stealing"
        rm -rf "$LOCK"; mkdir "$LOCK"; echo $$ > "$LOCK/pid"
        trap 'rm -rf "$LOCK"' EXIT
    else
        echo "Another runner holds the lock (pid $holder); exiting."
        exit 1
    fi
fi

SUMMARY="$LOG_ROOT/summary.tsv"

if [[ ! -f "$SUMMARY" ]]; then
    printf 'run_id\tdataset\tseq_len\tpred_len\tmaster_seed\tval_mse\tval_weighted_wtr\ttest_mse\ttest_mae\ttest_rmse\twtr5\twtr10\twtr15\tweighted_wtr\tstatus\tlog_path\n' > "$SUMMARY"
fi

extract_last() {
    sed -nE "$1" "$2" 2>/dev/null | tail -n 1
}

total=0
ok=0
for sl in $SEQLENS; do
    for pl in $PREDLENS; do
        total=$((total+1))
        run_id="${DATASET}_sl${sl}_h${pl}_seed${SEED}"
        run_dir="$LOG_ROOT/$run_id"
        log_file="$run_dir/run.log"
        done_file="$run_dir/DONE"
        ckpt="$run_dir/best.pt"
        if [[ -f "$done_file" ]]; then
            echo "SKIP $run_id"
            ok=$((ok+1))
            continue
        fi
        mkdir -p "$run_dir"
        echo "[$(date '+%F %T')] RUN $run_id"
        CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u "$MAIN" \
            --dataset "$DATASET" \
            --data_dir "$DATA_DIR" \
            --seq_len "$sl" \
            --pred_len "$pl" \
            --epochs 100 \
            --stage1_epochs 60 \
            --stage2_epochs 30 \
            --stage2_patience 5 \
            --stage3_patience 10 \
            --batch_size 32 \
            --lr 3e-5 \
            --stage1_lr 1e-4 \
            --stage2_lr 1e-4 \
            --finetune_lr 1e-5 \
            --stage1_fusion independent \
            --stage2_scope gate_only \
            --stage3_scope gate_only \
            --fusion_mode bounded_gate \
            --gate_mode horizon \
            --prior_weights 0.853,0.142,0.004 \
            --dynamic_blend 0.20 \
            --disable_entropy_state \
            --gate_init zero \
            --balance_weight 0 \
            --horizon_balance_weight 0 \
            --route_loss_weight 0 \
            --wtr_loss_weight 0 \
            --selection_metric weighted_wtr \
            --selection_mse_tolerance 0.01 \
            --min_epochs 15 \
            --patience 15 \
            --master_seed "$SEED" \
            --device auto \
            --trend_variant v2 \
            --periodic_variant v2 \
            --ramp_variant v1 \
            --save_best_checkpoint "$ckpt" \
            --suppress_horizon_weights > "$log_file" 2>&1
        exit_code=$?
        if [[ $exit_code -eq 0 ]]; then
            touch "$done_file"
            ok=$((ok+1))
        fi
        val_mse=$(extract_last "s/.*Selected Val:.* mse=([0-9.eE+-]+).*/\1/p" "$log_file")
        val_wtr=$(extract_last "s/.*Selected Val:.* weighted_wtr=([0-9.eE+-]+).*/\1/p" "$log_file")
        test_mse=$(extract_last "s/.*Test:.*'mse': ([0-9.eE+-]+).*/\1/p" "$log_file")
        test_mae=$(extract_last "s/.*Test:.*'mae': ([0-9.eE+-]+).*/\1/p" "$log_file")
        test_rmse=$(extract_last "s/.*Test:.*'rmse': ([0-9.eE+-]+).*/\1/p" "$log_file")
        wtr5=$(extract_last "s/.*Test:.*'wtr5': ([0-9.eE+-]+).*/\1/p" "$log_file")
        wtr10=$(extract_last "s/.*Test:.*'wtr10': ([0-9.eE+-]+).*/\1/p" "$log_file")
        wtr15=$(extract_last "s/.*Test:.*'wtr15': ([0-9.eE+-]+).*/\1/p" "$log_file")
        wwtr=$(extract_last "s/.*Test:.*'weighted_wtr': ([0-9.eE+-]+).*/\1/p" "$log_file")
        status="ok"; [[ $exit_code -ne 0 ]] && status="failed:$exit_code"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$run_id" "$DATASET" "$sl" "$pl" "$SEED" \
            "${val_mse:--}" "${val_wtr:--}" "${test_mse:--}" "${test_mae:--}" "${test_rmse:--}" \
            "${wtr5:--}" "${wtr10:--}" "${wtr15:--}" "${wwtr:--}" "$status" "$log_file" >> "$SUMMARY"
        echo "[$(date '+%F %T')] DONE $run_id status=$status test_mse=${test_mse:-?} wtr=${wwtr:-?}"
    done
done
echo "Finished: $ok/$total succeeded. Summary: $SUMMARY"
