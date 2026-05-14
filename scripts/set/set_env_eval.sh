#!/bin/bash
# python 3.10 & cuda 11.8

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

conda update -n base -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main --override-channels conda -y
conda clean -a -y
python -m pip install --upgrade pip
python -m pip cache purge

conda install -y -c https://conda.anaconda.org/pytorch -c https://conda.anaconda.org/nvidia \
  pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=11.8
conda install -y -c https://conda.anaconda.org/xformers xformers==0.0.28.post3
python -m pip install --no-deps https://github.com/vllm-project/vllm/releases/download/v0.7.2/vllm-0.7.2+cu118-cp38-abi3-manylinux1_x86_64.whl

conda env update -f ./scripts/set/environment_eval.yml

export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export FLASH_ATTN_CUDA_ARCHS=89
export MAX_JOBS="${MAX_JOBS:-4}"
export NVCC_THREADS="${NVCC_THREADS:-4}"
python -m pip install packaging ninja
python -m pip install flash-attn==2.7.3 --no-build-isolation --no-cache-dir
