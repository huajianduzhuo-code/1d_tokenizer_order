#!/bin/bash
# Run GenEval evaluation across all (model, order, seed) combinations.
# 8 GPUs per task; one task at a time. After each parallel detection run,
# parse results.jsonl into geneval_summary.json (with seed field).
set -e

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate geneval

cd /home/hliu256/1d-tokenizer
BASE=/data3/haoyuliu/geneval_eval
SEEDS=${SEEDS:-"42 43 44"}
ORDERS=${ORDERS:-"random prompt_sim prompt_sim_rev left_to_right right_to_left center_out"}
MODELS=${MODELS:-"l xl"}

echo "============================================"
echo " GenEval Evaluation"
echo "   seeds:  $SEEDS"
echo "   models: $MODELS"
echo "   orders: $ORDERS"
echo "============================================"

for SEED in $SEEDS; do
    for MODEL in $MODELS; do
        for ORDER in $ORDERS; do
            DIR=$BASE/maskgen_kl_${MODEL}_${ORDER}_seed${SEED}
            RESULTS=$DIR/results.jsonl
            SUMMARY=$DIR/geneval_summary.json
            if [ -f "$SUMMARY" ]; then
                echo "--- $MODEL / $ORDER / seed=$SEED: summary exists, skipping ---"
                continue
            fi
            if [ ! -d "$DIR" ]; then
                echo "--- $MODEL / $ORDER / seed=$SEED: dir missing, skipping ---"
                continue
            fi
            if [ ! -f "$RESULTS" ]; then
                echo ""
                echo "--- Detecting $MODEL / $ORDER / seed=$SEED (8 GPUs) ---"
                python eval_geneval_parallel.py "$DIR" --num-gpus 8
            fi
            # Parse results.jsonl -> geneval_summary.json (lightweight, runs in
            # geneval env which has pandas).
            python eval_geneval.py --model "$MODEL" --order "$ORDER" --seed "$SEED" --save-summary
        done
    done
done

# Print final mean±std summary in maskgen env (pandas + numpy needed).
conda activate maskgen
echo ""
echo "============================================"
echo " Summary"
echo "============================================"
python eval_geneval.py --summary
