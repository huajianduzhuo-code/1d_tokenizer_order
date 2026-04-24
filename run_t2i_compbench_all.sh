#!/bin/bash
# Generate T2I-CompBench++ val images for all 4 {model, order} configs in parallel.
# 8x GPU -> 2 GPUs per config, 4 configs in parallel. Evaluation runs in a
# separate env; see run_t2i_compbench_evaluate.sh.
set -e

cd /home/hliu256/1d-tokenizer
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate maskgen

echo "============================================================"
echo " T2I-CompBench++: Launching 4 configs in parallel"
echo " (8 categories x 10 samples/prompt x ~300 prompts/cat = ~24k"
echo "  images/config; 4 configs -> ~96k images total)"
echo "============================================================"

CUDA_VISIBLE_DEVICES=0,1 python eval_t2i_compbench.py --model l  --order random     --num-gpus 2 --category all &
PID1=$!
CUDA_VISIBLE_DEVICES=2,3 python eval_t2i_compbench.py --model l  --order prompt_sim --num-gpus 2 --category all &
PID2=$!
CUDA_VISIBLE_DEVICES=4,5 python eval_t2i_compbench.py --model xl --order random     --num-gpus 2 --category all &
PID3=$!
CUDA_VISIBLE_DEVICES=6,7 python eval_t2i_compbench.py --model xl --order prompt_sim --num-gpus 2 --category all &
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
echo "============================================================"
echo " Generation complete!"
echo " Output: /data3/haoyuliu/t2i_compbench_eval/"
echo "============================================================"
echo ""
echo "Next: conda activate /data3/haoyuliu/conda_envs/t2i_compbench"
echo "      bash run_t2i_compbench_evaluate.sh"
