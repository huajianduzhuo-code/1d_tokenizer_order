#!/bin/bash
# Run DPG-Bench + GenEval on the newly added 1D static orders
# (left_to_right, right_to_left, center_out) and save logs/summaries.
#
# Tunables (env vars):
#   ORDERS  default: "left_to_right right_to_left center_out"
#   SEEDS   default: "42"
#   MODEL   default: "xl"            (single model -- generation step)
#   MODELS  default: "xl"            (eval step expects a space-separated list)
#   SKIP_GENEVAL=1 to skip GenEval, SKIP_DPG=1 to skip DPG.
#
# Outputs:
#   /data3/haoyuliu/geneval_eval/maskgen_kl_<model>_<order>_seed<seed>/...
#   /data3/haoyuliu/dpg_bench_eval/maskgen_kl_<model>_<order>_seed<seed>/...
#   /data3/haoyuliu/ablation_logs/<timestamp>/{geneval,dpg}_{generate,evaluate,summary}.log
#   /data3/haoyuliu/ablation_logs/<timestamp>/geneval_summary_table.txt
#   /data3/haoyuliu/ablation_logs/<timestamp>/dpg_summary_tables.txt

set -e

cd /home/hliu256/1d-tokenizer

ORDERS=${ORDERS:-"left_to_right right_to_left center_out"}
SEEDS=${SEEDS:-"42"}
MODEL=${MODEL:-"xl"}
MODELS=${MODELS:-"$MODEL"}

TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR=/data3/haoyuliu/ablation_logs/$TS
mkdir -p "$LOG_DIR"

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

echo "============================================"
echo " New-orders ablation"
echo "   timestamp: $TS"
echo "   ORDERS:    $ORDERS"
echo "   SEEDS:     $SEEDS"
echo "   MODEL:     $MODEL  (MODELS for eval: $MODELS)"
echo "   logs:      $LOG_DIR"
echo "============================================"

run_step () {
    # run_step <log_name> -- <command...>
    local name=$1; shift
    [[ "$1" == "--" ]] && shift
    local log="$LOG_DIR/${name}.log"
    echo ""
    echo "---- [$name] $* ----"
    echo "     log: $log"
    if "$@" 2>&1 | tee "$log"; then
        echo "---- [$name] OK ----"
    else
        local rc=${PIPESTATUS[0]}
        echo "---- [$name] FAILED (rc=$rc) ----"
        return "$rc"
    fi
}

# ── GenEval ───────────────────────────────────────────────────────────────
if [[ -z "$SKIP_GENEVAL" ]]; then
    echo ""
    echo "############ GenEval ############"

    ORDERS="$ORDERS" SEEDS="$SEEDS" MODEL="$MODEL" \
        run_step geneval_generate -- bash run_geneval_orders.sh

    ORDERS="$ORDERS" SEEDS="$SEEDS" MODELS="$MODELS" \
        run_step geneval_evaluate -- bash run_geneval_evaluate.sh

    conda activate maskgen
    run_step geneval_summary -- python eval_geneval.py --summary
    cp -f /data3/haoyuliu/geneval_eval/geneval_summary_table.txt "$LOG_DIR/" 2>/dev/null || true
fi

# ── DPG-Bench ─────────────────────────────────────────────────────────────
if [[ -z "$SKIP_DPG" ]]; then
    echo ""
    echo "############ DPG-Bench ############"

    ORDERS="$ORDERS" SEEDS="$SEEDS" MODEL="$MODEL" \
        run_step dpg_generate -- bash run_dpg_orders.sh

    ORDERS="$ORDERS" SEEDS="$SEEDS" MODELS="$MODELS" \
        run_step dpg_evaluate -- bash run_dpg_evaluate.sh

    conda activate maskgen
    run_step dpg_summary -- python eval_dpg.py --summary
    cp -f /data3/haoyuliu/dpg_bench_eval/dpg_summary_tables.txt "$LOG_DIR/" 2>/dev/null || true
fi

echo ""
echo "============================================"
echo " Ablation complete."
echo "   logs:     $LOG_DIR"
echo "   GenEval:  /data3/haoyuliu/geneval_eval/"
echo "   DPG:      /data3/haoyuliu/dpg_bench_eval/"
echo "============================================"
