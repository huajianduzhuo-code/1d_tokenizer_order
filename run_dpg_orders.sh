#!/bin/bash
# Generate DPG-Bench images for an arbitrary list of orders.
#
# Usage:
#   ORDERS="left_to_right right_to_left center_out random" \
#       SEEDS="42 43 44" MODEL=xl bash run_dpg_orders.sh
#
# Defaults: ORDERS="random prompt_sim left_to_right right_to_left center_out",
#           SEEDS="42", MODEL=xl, NUM_GPUS_PER_JOB=2, TOTAL_GPUS=8.
#
# After generation, evaluate with run_dpg_evaluate.sh (DPG conda env).
set -e

cd /home/hliu256/1d-tokenizer
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate maskgen

ORDERS=${ORDERS:-"random prompt_sim left_to_right right_to_left center_out"}
SEEDS=${SEEDS:-"42"}
MODEL=${MODEL:-"xl"}
NUM_GPUS_PER_JOB=${NUM_GPUS_PER_JOB:-2}
TOTAL_GPUS=${TOTAL_GPUS:-8}
JOBS_PER_BATCH=$(( TOTAL_GPUS / NUM_GPUS_PER_JOB ))

echo "ORDERS:  $ORDERS"
echo "SEEDS:   $SEEDS"
echo "MODEL:   $MODEL"
echo "GPUs/job=$NUM_GPUS_PER_JOB  parallel jobs/batch=$JOBS_PER_BATCH"
echo ""

read -ra ORDER_ARR <<< "$ORDERS"
NUM_ORDERS=${#ORDER_ARR[@]}

for SEED in $SEEDS; do
    echo "============================================"
    echo " DPG-Bench seed=$SEED  ($NUM_ORDERS orders)"
    echo "============================================"

    for (( i=0; i<NUM_ORDERS; i+=JOBS_PER_BATCH )); do
        PIDS=()
        for (( j=0; j<JOBS_PER_BATCH && (i+j)<NUM_ORDERS; j++ )); do
            ORDER=${ORDER_ARR[$((i+j))]}
            START=$((j * NUM_GPUS_PER_JOB))
            END=$((START + NUM_GPUS_PER_JOB - 1))
            DEVS=$(seq -s, $START $END)
            echo "  -> CUDA_VISIBLE_DEVICES=$DEVS  order=$ORDER  model=$MODEL  seed=$SEED"
            CUDA_VISIBLE_DEVICES=$DEVS python eval_dpg.py \
                --model "$MODEL" --order "$ORDER" \
                --num-gpus "$NUM_GPUS_PER_JOB" --seed "$SEED" &
            PIDS+=($!)
        done
        FAIL=0
        for PID in "${PIDS[@]}"; do
            wait "$PID" || FAIL=$((FAIL+1))
        done
        if [ "$FAIL" -ne 0 ]; then
            echo "WARNING: $FAIL job(s) failed in this batch (seed=$SEED, batch starting at order index $i)"
        fi
    done
done

echo ""
echo "============================================"
echo " Generation complete."
echo " Output: /data3/haoyuliu/dpg_bench_eval/"
echo " Next:   conda activate /data3/haoyuliu/conda_envs/dpg && bash run_dpg_evaluate.sh"
echo "============================================"
