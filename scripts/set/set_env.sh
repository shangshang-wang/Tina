#!/bin/bash
# python 3.10 + cuda 11.8.0

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

conda update -n base -c defaults conda -y
conda clean -a -y
pip install --upgrade pip
pip cache purge

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
pip install xformers==0.0.28.post3 --index-url https://download.pytorch.org/whl/cu118
pip install https://github.com/vllm-project/vllm/releases/download/v0.7.2/vllm-0.7.2+cu118-cp38-abi3-manylinux1_x86_64.whl

conda env update -f ./scripts/set/environment.yml
