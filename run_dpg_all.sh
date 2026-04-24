#!/bin/bash
# Generate DPG-Bench images for all 4 {model, order} configs in parallel.
# 8x A5000 -> 2 GPUs per config, 4 configs in parallel.
# Evaluation runs in a separate env; see run_dpg_evaluate.sh.

set -e

cd /home/hliu256/1d-tokenizer
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate maskgen

echo "============================================"
echo " DPG-Bench: Launching 4 configs in parallel"
echo "============================================"

CUDA_VISIBLE_DEVICES=0,1 python eval_dpg.py --model l  --order random      --num-gpus 2 &
PID1=$!
CUDA_VISIBLE_DEVICES=2,3 python eval_dpg.py --model l  --order prompt_sim  --num-gpus 2 &
PID2=$!
CUDA_VISIBLE_DEVICES=4,5 python eval_dpg.py --model xl --order random      --num-gpus 2 &
PID3=$!
CUDA_VISIBLE_DEVICES=6,7 python eval_dpg.py --model xl --order prompt_sim  --num-gpus 2 &
PID4=$!

echo "PIDs: L-random=$PID1  L-prompt_sim=$PID2  XL-random=$PID3  XL-prompt_sim=$PID4"
echo "Waiting for all generation jobs..."

FAIL=0
for PID in $PID1 $PID2 $PID3 $PID4; do
    wait $PID || FAIL=$((FAIL+1))
done

if [ $FAIL -ne 0 ]; then
    echo "WARNING: $FAIL generation job(s) failed"
fi

echo ""
echo "============================================"
echo " Generation complete!"
echo " Output: /data3/haoyuliu/dpg_bench_eval/"
echo "============================================"
echo ""
echo "Next: conda activate dpg && bash run_dpg_evaluate.sh"
