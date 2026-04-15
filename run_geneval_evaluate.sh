#!/bin/bash
# Run GenEval evaluation on all generated images using 8 GPUs in parallel.
set -e

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate geneval

cd /home/hliu256/1d-tokenizer
BASE=/data3/haoyuliu/geneval_eval

echo "============================================"
echo " GenEval Evaluation (8 GPUs per task)"
echo "============================================"

for MODEL in l xl; do
    for ORDER in random prompt_sim; do
        RESULTS=$BASE/maskgen_kl_${MODEL}_${ORDER}/results.jsonl
        if [ -f "$RESULTS" ]; then
            echo "--- MaskGen-KL-${MODEL^^} / $ORDER: already done, skipping ---"
        else
            echo ""
            echo "--- Evaluating MaskGen-KL-${MODEL^^} / $ORDER (8 GPUs) ---"
            python eval_geneval_parallel.py $BASE/maskgen_kl_${MODEL}_${ORDER} --num-gpus 8
        fi
        echo ""
    done
done

# Summary
conda activate maskgen
echo "============================================"
echo " Summary"
echo "============================================"
python eval_geneval.py --summary
