#!/bin/bash
# python 3.11 & cuda 11.8

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

conda clean -a -y
mamba clean -a -y
pip install --upgrade pip
pip cache purge

mamba install cuda -c nvidia/label/cuda-11.8.0 -y
mamba install gcc gxx -c conda-forge -y
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118

pip install xformers==0.0.28.post3 --index-url https://download.pytorch.org/whl/cu118
pip install https://github.com/vllm-project/vllm/releases/download/v0.7.2/vllm-0.7.2+cu118-cp38-abi3-manylinux1_x86_64.whl
pip install flash-attn==2.7.3 --no-build-isolation

## copied from https://github.com/huggingface/open-r1/blob/main/setup.py
pip install accelerate==1.4.0
pip install datasets>=3.2.0
pip install deepspeed==0.15.4
pip install distilabel[vllm,ray,openai]>=1.5.2
pip install e2b-code-interpreter>=1.0.5
pip install einops>=0.8.0
pip install flake8>=6.0.0
pip install huggingface_hub
pip install hf_transfer>=0.1.4
pip install isort>=5.12.0
pip install langdetect # Needed for LightEval's extended tasks
pip install latex2sympy2_extended>=1.0.6
pip install liger_kernel==0.5.3
GIT_LFS_SKIP_SMUDGE=1 pip install "lighteval @ git+https://github.com/huggingface/lighteval.git@ed084813e0bd12d82a06d9f913291fdbee774905"
pip install math-verify==0.5.2
pip install packaging>=23.0
pip install parameterized>=0.9.0
pip install peft>=0.14.0
pip install pytest
pip install python-dotenv
pip install ruff>=0.9.0
pip install safetensors>=0.3.3
pip install sentencepiece>=0.1.99
pip install transformers==4.49.0
pip install vllm==0.7.2
pip install "trl @ git+https://github.com/huggingface/trl.git@69ad852e5654a77f1695eb4c608906fe0c7e8624"
pip install wandb
