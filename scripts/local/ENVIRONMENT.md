# Tina local environment

This machine uses one conda environment for both Tina training and evaluation:

```bash
conda activate tina
cd /home/wangls/Tina
source ./scripts/set/set_vars.sh
```

Core stack:

- Python 3.10
- PyTorch 2.5.1
- CUDA runtime/toolchain 11.8 inside the conda environment
- xformers 0.0.28.post3 from conda
- flash-attn 2.7.3 built locally against conda CUDA 11.8
- vLLM 0.7.2
- TRL 0.15.2
- Transformers 4.50.0
- PEFT 0.15.0
- DeepSpeed 0.16.4
- LightEval 0.8.1

The repository YAML files were not applied directly. They freeze too many transitive
packages, include the unstable `pkgs/r` channel, and mix installation concerns with
base-conda updates. The local environment was built from the repo imports and
script entry points instead.

Important build details:

```bash
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export CPATH="$CONDA_PREFIX/targets/x86_64-linux/include:${CPATH:-}"
export CPLUS_INCLUDE_PATH="$CONDA_PREFIX/targets/x86_64-linux/include:${CPLUS_INCLUDE_PATH:-}"
export FLASH_ATTN_CUDA_ARCHS=80
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export MAX_JOBS=4
export NVCC_THREADS=4
```

`CPATH` / `CPLUS_INCLUDE_PATH` are needed because the conda CUDA headers containing
`cuda/std/utility` live under `targets/x86_64-linux/include`, while flash-attn's
build command does not add that directory by default.

If rebuilding flash-attn later, avoid system `/usr/local/cuda-12`; this repo was
verified with conda CUDA 11.8 and RTX 3090 (`sm80`).
