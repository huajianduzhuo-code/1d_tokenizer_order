#!/bin/bash
# Evaluate all 4 DPG-Bench configs sequentially using the mPLUG VQA model.
# Must be run from the `dpg` conda env (see setup_dpg_env.sh).
# Each run uses accelerate + all visible GPUs.

set -e

cd /home/hliu256/1d-tokenizer
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
# dpg env lives on /data3 (see setup_dpg_env.sh) because /data1 is full.
conda activate /data3/haoyuliu/conda_envs/dpg

echo "============================================"
echo " DPG-Bench: sequential evaluation (4 configs)"
echo "============================================"

for ORDER in random prompt_sim; do
    for MODEL in l xl; do
        SUMMARY=/data3/haoyuliu/dpg_bench_eval/maskgen_kl_${MODEL}_${ORDER}/dpg_summary.json
        if [ -f "$SUMMARY" ]; then
            echo "Skipping $MODEL / $ORDER (summary exists)"
            continue
        fi
        echo ""
        echo "--- Evaluating $MODEL / $ORDER ---"
        python eval_dpg.py --model "$MODEL" --order "$ORDER" --eval-only
    done
done

echo ""
echo "============================================"
echo " All evaluations complete."
echo " Run: conda activate maskgen && python eval_dpg.py --summary"
echo "============================================"
