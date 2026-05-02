#!/bin/bash
# Create conda env for T2I-CompBench++ evaluation (BLIP-VQA, UniDet, CLIPScore, 3-in-1).
#
# T2I-CompBench pins an older stack (torch 2.0.1, transformers 4.30.2,
# diffusers 0.15.0.dev0, detectron2 from a specific commit) that conflicts
# with our maskgen env (torch 2.5.1) and the dpg env. We isolate it at:
#
#   /data3/haoyuliu/conda_envs/t2i_compbench
#
# This script is idempotent — rerunning skips finished steps when possible.
set -e

ENV_NAME=t2i_compbench
T2I_ROOT=/home/hliu256/T2I-CompBench
ENV_PREFIX=/data3/haoyuliu/conda_envs/$ENV_NAME
WEIGHTS_DIR=$T2I_ROOT/UniDet_eval/experts/expert_weights

# /data1 is full (see setup_dpg_env.sh notes). Redirect conda + pip caches to
# /data3 so big downloads (cuda-toolkit, etc.) don't hit ENOSPC.
export CONDA_PKGS_DIRS=/data3/haoyuliu/conda_pkgs
export PIP_CACHE_DIR=/data3/haoyuliu/pip_cache
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR"

echo "============================================================"
echo " T2I-CompBench env setup -> $ENV_PREFIX"
echo "============================================================"

# ---------- 1. conda env ----------
if [ ! -d "$ENV_PREFIX" ]; then
    echo "[1/6] Creating conda env (Python 3.10)..."
    mkdir -p "$(dirname $ENV_PREFIX)"
    conda create --prefix $ENV_PREFIX python=3.10 -y
else
    echo "[1/6] Conda env already exists, skipping create."
fi

PIP=$ENV_PREFIX/bin/pip
PY=$ENV_PREFIX/bin/python

# ---------- 2. torch 2.0.1 + cu118 (works with our 560.x driver) ----------
echo "[2/6] Installing torch 2.0.1 + cu118..."
$PIP install "numpy<2" torch==2.0.1 torchvision==0.15.2 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# Install matching CUDA 11.8 toolkit + gcc 11 via conda into the env. We
# need nvcc to build detectron2 from source (no prebuilt wheel for torch 2.0
# exists), and CUDA 11.8 officially caps host gcc at 11 — system gcc on this
# box is 13.
#
# Split into two conda calls so a post-link failure in one doesn't roll back
# the other (we hit exactly this with conda-forge's gdb when installing both
# in a single transaction). Use nvidia's own cu118 label for CUDA components;
# it bundles cuda-gdb but avoids pulling in conda-forge's `gdb` package.
echo "  Installing CUDA 11.8 nvcc + libs (nvidia channel)..."
conda install -p $ENV_PREFIX \
    -c nvidia/label/cuda-11.8.0 \
    cuda-nvcc cuda-cudart-dev cuda-cccl cuda-driver-dev \
    libcublas-dev libcusolver-dev libcurand-dev libcusparse-dev \
    libnpp-dev libcufft-dev -y

echo "  Installing gcc 11 (conda-forge)..."
conda install -p $ENV_PREFIX \
    -c conda-forge \
    gxx_linux-64=11.4.0 gcc_linux-64=11.4.0 -y

# ---------- 3. T2I-CompBench requirements (minus torch + detectron2 — handled separately) ----------
echo "[3/6] Installing T2I-CompBench requirements (excluding torch/detectron2)..."
REQ_SRC=$T2I_ROOT/requirements.txt
REQ_TRIMMED=/tmp/t2i_compbench_requirements_trimmed.txt
# Skip torch/torchvision (pinned above), detectron2 (built from source below),
# and the bundled nvidia-*-cu11 wheels (conflict with our cu118 torch).
grep -vE '^(torch==|torchvision==|git\+https://github\.com/facebookresearch/detectron2|nvidia-)' \
    "$REQ_SRC" > "$REQ_TRIMMED"
$PIP install -r "$REQ_TRIMMED"

# ---------- 4. detectron2 (UniDet dep) ----------
echo "[4/6] Installing detectron2 from pinned commit..."
# Point the build at the env's cu118 toolkit + gcc 11 (installed in step 2).
export CUDA_HOME=$ENV_PREFIX
export PATH=$ENV_PREFIX/bin:$PATH
export CC=$(ls $ENV_PREFIX/bin/x86_64-conda-linux-gnu-gcc 2>/dev/null || which gcc)
export CXX=$(ls $ENV_PREFIX/bin/x86_64-conda-linux-gnu-g++ 2>/dev/null || which g++)
echo "  CUDA_HOME=$CUDA_HOME"
echo "  CC=$CC"
echo "  CXX=$CXX"
echo "  nvcc -> $(which nvcc)"

# detectron2's setup.py imports torch at build time to pick the right CUDA
# version. With pip's default build isolation the setup.py runs in a fresh
# env without torch, so we must pass --no-build-isolation. Also pre-install
# setuptools + wheel + ninja so the build can run offline of the isolated env.
$PIP install -U "setuptools<70" wheel ninja
$PIP install --no-build-isolation \
    'git+https://github.com/facebookresearch/detectron2.git@5aeb252b194b93dc2879b4ac34bc51a31b5aee13' || {
    echo "ERROR: detectron2 build failed. Inspect the stderr above."
    exit 1
}

# ---------- 5. UniDet + MiDaS weights ----------
echo "[5/6] Downloading UniDet / MiDaS weights to $WEIGHTS_DIR ..."
mkdir -p "$WEIGHTS_DIR"
cd "$WEIGHTS_DIR"

if [ ! -f "Unified_learned_OCIM_RS200_6x+2x.pth" ]; then
    echo "  - UniDet RS200 6x+2x"
    wget -q "https://huggingface.co/shikunl/prismer/resolve/main/expert_weights/Unified_learned_OCIM_RS200_6x%2B2x.pth" \
        -O "Unified_learned_OCIM_RS200_6x+2x.pth"
fi
if [ ! -f "dpt_hybrid-midas-501f0c75.pt" ]; then
    echo "  - MiDaS dpt_hybrid"
    wget -q "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/dpt_hybrid-midas-501f0c75.pt"
fi
# Google Drive weight (required per T2I-CompBench Readme §2). File id from Readme.
if ! ls Unified_learned_OCIM_R50_6x*.pth >/dev/null 2>&1; then
    echo "  - UniDet R50 (gdown)"
    $PIP install -q gdown
    $ENV_PREFIX/bin/gdown "https://docs.google.com/uc?id=1C4sgkirmgMumKXXiLOPmCKNTZAc3oVbq" || {
        echo "  WARN: gdown failed. Download manually from:"
        echo "    https://docs.google.com/uc?id=1C4sgkirmgMumKXXiLOPmCKNTZAc3oVbq"
        echo "    and place into $WEIGHTS_DIR"
    }
fi
cd /home/hliu256/1d-tokenizer

# ---------- 6. accelerate default config + spacy model ----------
echo "[6/6] accelerate default config + spacy model..."
$PY -c "from accelerate.utils import write_basic_config; write_basic_config(mixed_precision='fp16')"
# en_core_web_sm is already in requirements.txt as a wheel URL, but double-check.
$PY -c "import spacy; spacy.load('en_core_web_sm')" || \
    $PY -m spacy download en_core_web_sm

echo ""
echo "============================================================"
echo " Done."
echo " Activate with:  conda activate $ENV_PREFIX"
echo " Then evaluate:  python eval_t2i_compbench.py --model xl --order prompt_sim --eval-only"
echo "============================================================"
