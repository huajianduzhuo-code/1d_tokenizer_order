#!/bin/bash
# Run all 4 PromptBind-Hard configs × multiple seeds.
# Phase 1: GPU-parallel generation (4 configs in parallel, 2 GPUs each).
# Phase 2: sequential VLM judging (API-bound, parallel runs would trip rate limits).
#
# Env knobs (all optional):
#   SEEDS="42 43 44"         seeds to sweep
#   JUDGE_PROVIDER=openai    openai | gemini | both
#   DATASET=benchmarks/pbh/prompts.jsonl   path to prompts JSONL (changes
#                                          output base via _output_base_for_dataset)
#
# Required env: OPENAI_API_KEY (and/or GOOGLE_API_KEY) for the judging phase.

set -e

cd /home/hliu256/1d-tokenizer
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate maskgen

SEEDS=${SEEDS:-"42 43 44"}
JUDGE_PROVIDER=${JUDGE_PROVIDER:-"openai"}
DATASET=${DATASET:-"benchmarks/pbh/prompts.jsonl"}
DATASET_ARGS="--dataset $DATASET"
echo ">>> dataset: $DATASET"

# -----------------------------------------------------------
# Phase 1: Generation (GPU-parallel)
# -----------------------------------------------------------
for SEED in $SEEDS; do
    echo "============================================"
    echo " PBH gen: 4 configs in parallel  (seed=$SEED)"
    echo "============================================"

    CUDA_VISIBLE_DEVICES=0,1 python eval_pbh.py --model l  --order random     --num-gpus 2 --seed $SEED $DATASET_ARGS &
    PID1=$!
    CUDA_VISIBLE_DEVICES=2,3 python eval_pbh.py --model l  --order prompt_sim --num-gpus 2 --seed $SEED $DATASET_ARGS &
    PID2=$!
    CUDA_VISIBLE_DEVICES=4,5 python eval_pbh.py --model xl --order random     --num-gpus 2 --seed $SEED $DATASET_ARGS &
    PID3=$!
    CUDA_VISIBLE_DEVICES=6,7 python eval_pbh.py --model xl --order prompt_sim --num-gpus 2 --seed $SEED $DATASET_ARGS &
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
echo "============================================"
echo " Generation complete (seeds: $SEEDS)"
echo " Output: /data3/haoyuliu/pbh_eval/"
echo "============================================"
echo ""

# -----------------------------------------------------------
# Phase 2: Judging (sequential, API-bound)
# -----------------------------------------------------------
echo "============================================"
echo " PBH judge: provider=$JUDGE_PROVIDER (sequential)"
echo "============================================"

if [ "$JUDGE_PROVIDER" = "openai" ] || [ "$JUDGE_PROVIDER" = "both" ]; then
    if [ -z "$OPENAI_API_KEY" ]; then
        echo "ERROR: OPENAI_API_KEY not set; export it before running judging."
        exit 1
    fi
fi
if [ "$JUDGE_PROVIDER" = "gemini" ] || [ "$JUDGE_PROVIDER" = "both" ]; then
    if [ -z "$GOOGLE_API_KEY" ]; then
        echo "ERROR: GOOGLE_API_KEY not set; export it before running judging."
        exit 1
    fi
fi

for SEED in $SEEDS; do
    for MODEL in l xl; do
        for ORDER in random prompt_sim; do
            echo "--- judge: $MODEL / $ORDER / seed=$SEED ---"
            python eval_pbh.py --model $MODEL --order $ORDER --seed $SEED \
                $DATASET_ARGS --eval-only --judge-provider $JUDGE_PROVIDER || \
                echo "WARNING: judging failed for $MODEL/$ORDER/seed=$SEED"
        done
    done
done

# -----------------------------------------------------------
# Cross-config summary table
# -----------------------------------------------------------
echo ""
echo "============================================"
echo " PBH summary table"
echo "============================================"
python eval_pbh.py --summary $DATASET_ARGS
