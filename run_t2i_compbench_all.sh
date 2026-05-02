#!/bin/bash
# Generate T2I-CompBench++ val images for all 4 {model, order} configs × multiple seeds.
# 8x GPU -> 2 GPUs per config, 4 configs in parallel inside each seed.
# Default seeds: 42 43 44 (override with $SEEDS).
# Evaluation runs in a separate env; see run_t2i_compbench_evaluate.sh.
set -e

cd /home/hliu256/1d-tokenizer
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate maskgen

SEEDS=${SEEDS:-"42 43 44"}

for SEED in $SEEDS; do
    echo "============================================================"
    echo " T2I-CompBench++: 4 configs in parallel  (seed=$SEED)"
    echo " (8 cats × 10 samples × ~300 prompts = ~24k images/config)"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES=0,1 python eval_t2i_compbench.py --model l  --order random     --num-gpus 2 --category all --seed $SEED &
    PID1=$!
    CUDA_VISIBLE_DEVICES=2,3 python eval_t2i_compbench.py --model l  --order prompt_sim --num-gpus 2 --category all --seed $SEED &
    PID2=$!
    CUDA_VISIBLE_DEVICES=4,5 python eval_t2i_compbench.py --model xl --order random     --num-gpus 2 --category all --seed $SEED &
    PID3=$!
    CUDA_VISIBLE_DEVICES=6,7 python eval_t2i_compbench.py --model xl --order prompt_sim --num-gpus 2 --category all --seed $SEED &
    PID4=$!

    echo "PIDs (seed=$SEED): L-random=$PID1  L-prompt_sim=$PID2  XL-random=$PID3  XL-prompt_sim=$PID4"

    FAIL=0
    for PID in $PID1 $PID2 $PID3 $PID4; do
        wait $PID || FAIL=$((FAIL+1))
    done

    if [ $FAIL -ne 0 ]; then
        echo "WARNING: $FAIL generation job(s) failed at seed=$SEED"
    fi
done

echo ""
echo "============================================================"
echo " Generation complete (seeds: $SEEDS)"
echo " Output: /data3/haoyuliu/t2i_compbench_eval/"
echo "============================================================"
echo ""
echo "Next: conda activate /data3/haoyuliu/conda_envs/t2i_compbench"
echo "      bash run_t2i_compbench_evaluate.sh"
