#!/bin/bash
# Run MJHQ-30K FID for all 4 {model, order} configs × multiple seeds.
# 8x GPU -> 2 GPUs per config, 4 configs in parallel inside each seed.
# eval_mjhq_fid.py does generation + FID compute in one process.
# Default seeds: 0 1 2 (override with $SEEDS).
set -e

cd /home/hliu256/1d-tokenizer
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate maskgen

SEEDS=${SEEDS:-"0 1 2"}

for SEED in $SEEDS; do
    echo "============================================================"
    echo " MJHQ-30K FID: 4 configs in parallel  (seed=$SEED)"
    echo " (30k images per config × 4 configs = 120k images)"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES=0,1 python eval_mjhq_fid.py --model l  --order random     --num-gpus 2 --seed $SEED &
    PID1=$!
    CUDA_VISIBLE_DEVICES=2,3 python eval_mjhq_fid.py --model l  --order prompt_sim --num-gpus 2 --seed $SEED &
    PID2=$!
    CUDA_VISIBLE_DEVICES=4,5 python eval_mjhq_fid.py --model xl --order random     --num-gpus 2 --seed $SEED &
    PID3=$!
    CUDA_VISIBLE_DEVICES=6,7 python eval_mjhq_fid.py --model xl --order prompt_sim --num-gpus 2 --seed $SEED &
    PID4=$!

    echo "PIDs (seed=$SEED): L-random=$PID1  L-prompt_sim=$PID2  XL-random=$PID3  XL-prompt_sim=$PID4"

    FAIL=0
    for PID in $PID1 $PID2 $PID3 $PID4; do
        wait $PID || FAIL=$((FAIL+1))
    done

    if [ $FAIL -ne 0 ]; then
        echo "WARNING: $FAIL job(s) failed at seed=$SEED"
    fi
done

echo ""
echo "============================================================"
echo " All MJHQ-30K runs complete (seeds: $SEEDS)"
echo " Output: /data3/haoyuliu/mjhq30k_eval/"
echo "============================================================"
echo ""
echo "Print summary:"
echo "  python eval_mjhq_fid.py --summary --summary-suffix 30k"
