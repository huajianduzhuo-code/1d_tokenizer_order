#!/bin/bash
# Create a minimal conda environment for GenEval evaluation.
# Only needs: mmdet (Mask2Former) + open_clip + clip_benchmark
# Does NOT need maskgen/tatitok (generation is already done).
set -e

ENV_NAME=geneval
GENEVAL_ROOT=/home/hliu256/geneval

echo "Creating conda env: $ENV_NAME (Python 3.9 + PyTorch 1.12 + CUDA 11.3)"
conda create -n $ENV_NAME python=3.9 pytorch=1.12.1 torchvision=0.13.1 cudatoolkit=11.3 -c pytorch -c nvidia -y

PIP=/data1/haoyuliu/misc/miniconda3/envs/$ENV_NAME/bin/pip

echo "Installing mmcv-full + mmdet..."
$PIP install openmim
/data1/haoyuliu/misc/miniconda3/envs/$ENV_NAME/bin/mim install mmcv-full==1.7.1
$PIP install mmdet==2.28.2

echo "Installing open_clip + clip_benchmark..."
$PIP install open-clip-torch clip-benchmark pandas

echo "Installing mmdetection from source (2.x branch)..."
cd $GENEVAL_ROOT
if [ ! -d mmdetection ]; then
    git clone https://github.com/open-mmlab/mmdetection.git
    cd mmdetection && git checkout 2.x
else
    cd mmdetection
fi
$PIP install -e .

echo "Downloading Mask2Former weights..."
WEIGHT_DIR=/data3/haoyuliu/geneval_eval/mask2former_weights
mkdir -p $WEIGHT_DIR
if [ ! -f "$WEIGHT_DIR/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth" ]; then
    bash $GENEVAL_ROOT/evaluation/download_models.sh $WEIGHT_DIR
fi

echo ""
echo "Done! Test with:"
echo "  conda activate $ENV_NAME"
echo "  python -c 'import mmdet; print(mmdet.__version__)'"
