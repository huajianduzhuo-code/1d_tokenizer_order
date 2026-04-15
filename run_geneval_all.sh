#!/bin/bash
# Run all 4 GenEval configurations in parallel (generation), then evaluate.
# 8x A5000 → 2 GPUs per config, 4 configs in parallel.

set -e

cd /home/hliu256/1d-tokenizer
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate maskgen

echo "============================================"
echo " GenEval: Launching 4 configs in parallel"
echo "============================================"

# Generation: 4 configs x 2 GPUs each
CUDA_VISIBLE_DEVICES=0,1 python eval_geneval.py --model l  --order random     --num-gpus 2 &
PID1=$!
CUDA_VISIBLE_DEVICES=2,3 python eval_geneval.py --model l  --order prompt_sim  --num-gpus 2 &
PID2=$!
CUDA_VISIBLE_DEVICES=4,5 python eval_geneval.py --model xl --order random      --num-gpus 2 &
PID3=$!
CUDA_VISIBLE_DEVICES=6,7 python eval_geneval.py --model xl --order prompt_sim   --num-gpus 2 &
PID4=$!

echo "PIDs: L-random=$PID1  L-prompt_sim=$PID2  XL-random=$PID3  XL-prompt_sim=$PID4"
echo "Waiting for all generation jobs..."

# Wait and report
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
echo " Output: /data3/haoyuliu/geneval_eval/"
echo "============================================"
echo ""
echo "Next: install mmdet for evaluation, then run:"
echo "  bash run_geneval_evaluate.sh"
