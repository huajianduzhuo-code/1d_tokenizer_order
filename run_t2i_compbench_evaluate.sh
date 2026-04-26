#!/bin/bash
# Run T2I-CompBench++ evaluation for all 4 {model, order} × multiple seeds.
#
# Parallelism: configs and seeds run sequentially; within each config the
# 8 categories run sequentially; within each category the evaluator is
# image-sharded across all 8 GPUs (see _run_sharded_eval in
# eval_t2i_compbench.py). Default seeds: 42 43 44 (override with $SEEDS).
#
# Assumes image generation already finished via run_t2i_compbench_all.sh.
set -e

cd /home/hliu256/1d-tokenizer
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate /data3/haoyuliu/conda_envs/t2i_compbench

# Make all 8 GPUs visible to the driver; eval_t2i_compbench.py forks
# 8 subprocesses each pinned to one GPU via CUDA_VISIBLE_DEVICES=<i>.
unset CUDA_VISIBLE_DEVICES

SEEDS=${SEEDS:-"42 43 44"}

for SEED in $SEEDS; do
    for MODEL in l xl; do
        for ORDER in random prompt_sim; do
            echo ""
            echo "============================================================"
            echo " Eval: MaskGen-KL-${MODEL^^} / ${ORDER} / seed=${SEED}"
            echo " 8 categories sequential; each category sharded across 8 GPUs"
            echo "============================================================"
            python -u eval_t2i_compbench.py --model "$MODEL" --order "$ORDER" \
                --seed "$SEED" --category all --eval-only --num-gpus 8
        done
    done
done

echo ""
echo "============================================================"
echo " All evaluations done. Printing comparison table (mean±std)..."
echo "============================================================"
python -u eval_t2i_compbench.py --summary
