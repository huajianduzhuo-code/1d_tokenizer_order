#!/bin/bash
# Create a minimal conda environment for DPG-Bench evaluation (mPLUG VQA).
#
# This script was debugged against the DPG-Bench repo at /home/hliu256/ELLA on
# 2026-04-23. modelscope's VQA pipeline eagerly imports the OFA model, which
# pulls in a surprisingly large transitive closure (fairseq, torchaudio,
# librosa, etc). The version matrix below is the working combination.
set -e

ENV_NAME=dpg
ELLA_ROOT=/home/hliu256/ELLA
# /data1 is full, so install the env to /data3 (see CLAUDE.md).
ENV_PREFIX=/data3/haoyuliu/conda_envs/$ENV_NAME
MODELSCOPE_CACHE=/data3/haoyuliu/modelscope_cache

echo "Creating conda env at: $ENV_PREFIX (Python 3.10)"
mkdir -p "$(dirname $ENV_PREFIX)"
conda create --prefix $ENV_PREFIX python=3.10 -y

PIP=$ENV_PREFIX/bin/pip
PY=$ENV_PREFIX/bin/python

echo "Installing torch 2.5.1 + cu121 (driver is 560.x, matches maskgen env)..."
# Pin numpy<2 because torch 2.5.1 is compiled against NumPy 1.x.
$PIP install "numpy<2" torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --extra-index-url https://download.pytorch.org/whl/cu121

echo "Installing DPG-Bench requirements (excluding fairseq — handled below)..."
REQ_SRC=$ELLA_ROOT/requirements-for-dpg_bench.txt
REQ_TRIMMED=/tmp/dpg_requirements_trimmed.txt
grep -vE '^fairseq$' $REQ_SRC > $REQ_TRIMMED
$PIP install -r $REQ_TRIMMED

echo "Pinning transformers<5 (modelscope's OFA needs BasicTokenizer from 4.x)..."
$PIP install "transformers<5" "tokenizers<0.21"

echo "Installing fairseq from GitHub with --no-deps (pypi sdist is broken,"
echo "and its declared deps conflict with modern hydra/omegaconf)..."
$PIP install --no-deps "git+https://github.com/facebookresearch/fairseq.git@v0.12.2"

echo "Installing modelscope's undeclared runtime deps + fairseq runtime deps..."
# addict/simplejson/datasets/sortedcontainers/einops: modelscope eager imports
# hydra-core/sacrebleu/bitarray: needed by fairseq at import time
$PIP install \
    addict simplejson datasets sortedcontainers einops \
    sentencepiece tensorboardX easydict protobuf \
    hydra-core sacrebleu bitarray

echo "Pre-downloading mPLUG VQA checkpoint to $MODELSCOPE_CACHE ..."
mkdir -p $MODELSCOPE_CACHE
MODELSCOPE_CACHE=$MODELSCOPE_CACHE $PY - <<'PY'
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
pipe = pipeline(Tasks.visual_question_answering,
                model='damo/mplug_visual-question-answering_coco_large_en',
                device='cpu')
print("mPLUG VQA checkpoint ready.")
PY

echo ""
echo "Done. Activate with: conda activate $ENV_PREFIX"
echo "Then run evaluation with: python eval_dpg.py --model xl --order prompt_sim --eval-only"
