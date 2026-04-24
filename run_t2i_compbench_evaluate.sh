#!/bin/bash
# Run T2I-CompBench++ evaluation for all 4 {model, order} configs.
#
# Parallelism scheme: 4 configs run sequentially; within each config the 8
# categories run sequentially; within each category the evaluator is
# image-sharded across all 8 GPUs (see _run_sharded_eval in
# eval_t2i_compbench.py). So every per-category step uses 8 GPUs with
# roughly-balanced load and no idle-wait.
#
# Assumes image generation already finished via run_t2i_compbench_all.sh.
set -e

cd /home/hliu256/1d-tokenizer
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate /data3/haoyuliu/conda_envs/t2i_compbench

# Make all 8 GPUs visible to the driver script; eval_t2i_compbench.py forks
# 8 subprocesses each pinned to one GPU via CUDA_VISIBLE_DEVICES=<i>.
unset CUDA_VISIBLE_DEVICES

for MODEL in l xl; do
    for ORDER in random prompt_sim; do
        echo ""
        echo "============================================================"
        echo " Eval: MaskGen-KL-${MODEL^^} / ${ORDER}"
        echo " 8 categories sequential; each category sharded across 8 GPUs"
        echo "============================================================"
        python -u eval_t2i_compbench.py --model "$MODEL" --order "$ORDER" \
            --category all --eval-only --num-gpus 8
    done
done

echo ""
echo "============================================================"
echo " All evaluations done. Printing comparison table..."
echo "============================================================"
python -u eval_t2i_compbench.py --summary
