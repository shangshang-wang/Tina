# Tina Environment Setup Notes

This document summarizes the environment issues encountered on this machine and the fixes that worked.

Machine assumptions used in this repo:

- Repo root: `/home/wangls/Tina`
- User home: `/home/wangls`
- Conda root: `/home/wangls/miniconda3`
- System CUDA in `.bashrc`: `/usr/local/cuda-12`
- Training/eval target CUDA stack: `PyTorch 2.5.1 + CUDA 11.8`

## 1. Repo path layout

The original `scripts/set/set_vars.sh` assumed the repo lived under an extra `rl-reasoning/Tina` directory layer.
That was not true locally; the repo is directly at `/home/wangls/Tina`.

Fix:

- `scripts/set/set_vars.sh` now derives `REPO_ROOT` from the script path.
- Output paths now resolve to the current repo directly:

```text
PROJECT_DIR=/home/wangls/Tina
CKPT_DIR=/home/wangls/Tina/ckpts
DATA_DIR=/home/wangls/Tina/datasets
OUTPUT_DIR=/home/wangls/Tina/outputs
LOGGING_DIR=/home/wangls/Tina/logs
```

If the repo is moved, `set_vars.sh` should continue to work without editing hardcoded prefixes.

## 2. Script execution permission

`./scripts/set/set_env.sh` originally failed with `Permission denied`.

Cause:

- The file did not have the executable bit set.

Fix:

```bash
chmod +x ./scripts/set/set_env.sh
chmod +x ./scripts/set/set_env_eval.sh
```

Running with `bash ./scripts/set/set_env.sh` also works even without execute permission.

## 3. Conda channel and SSL issues

Observed error pattern:

```text
Retrying ... SSLEOFError ...
/pkgs/r/noarch/repodata.json.zst
```

Cause:

- `defaults` in conda config expands to both:
  - `repo.anaconda.com/pkgs/main`
  - `repo.anaconda.com/pkgs/r`
- The environment YAML files also explicitly listed `repo.anaconda.com/pkgs/main` and `repo.anaconda.com/pkgs/r`.
- This caused conda to keep touching `pkgs/r`, which was unstable in this network setup.

Fixes applied:

- Removed `defaults` from the active user config.
- Added explicit mirror channels instead.
- Removed explicit `repo.anaconda.com/pkgs/r` from environment files.
- Changed the `conda update` command in setup scripts to use an explicit mirror and `--override-channels`.

Current relevant files:

- `/home/wangls/.condarc`
- `/home/wangls/miniconda3/.condarc`
- `scripts/set/environment.yml`
- `scripts/set/environment_eval.yml`
- `scripts/set/set_env.sh`
- `scripts/set/set_env_eval.sh`

Useful cleanup command after changing channels:

```bash
conda clean -i -y
```

## 4. Proxy configuration pitfall

`.bashrc` contained both proxy exports and immediate `unset` lines.

That means a new shell ended up with no active `http_proxy` / `https_proxy`, even if the top of `.bashrc` looked correct.

Symptoms:

- Conda, GitHub, and other downloads behaved inconsistently.
- Checking `env | rg 'proxy'` showed no active proxy variables.

Recommendation:

- Do not export a proxy and then unset it in the same shell init path.
- If needed, set proxy explicitly in the current shell before installation:

```bash
export http_proxy="http://HOST:PORT"
export https_proxy="http://HOST:PORT"
```

If a proxy is required long-term, clean up `.bashrc` so the final state is unambiguous.

## 5. PyTorch installation

Problem:

- `download.pytorch.org` was unreliable from this machine.

Original behavior:

- `set_env.sh` and `set_env_eval.sh` installed torch packages with:

```bash
pip install ... --index-url https://download.pytorch.org/whl/cu118
```

Fix:

- Both setup scripts now install PyTorch from conda instead:

```bash
conda install -y -c pytorch -c nvidia \
  pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=11.8
```

This avoids the PyTorch wheel URL entirely.

## 6. xformers installation

Problem:

- `xformers` was previously installed from the PyTorch wheel URL path.

Fix:

- Install from the conda `xformers` channel instead:

```bash
conda install -y -c xformers xformers==0.0.28.post3
```

The `py310 + cu118 + pyt2.5.1` build exists in the channel and matches the current stack.

## 7. flash-attn installation

Problem:

- `conda env update -f ...` triggered pip installation of `flash-attn==2.7.3`.
- That failed with:

```text
ModuleNotFoundError: No module named 'torch'
```

Cause:

- pip build isolation created a temporary build environment without torch.
- `flash-attn` imports torch during build metadata generation.

Resolution:

- Install `flash-attn` only after torch is already installed.
- Use `--no-build-isolation`.
- Use the conda environment's CUDA 11.8, not system CUDA 12.

Working pattern:

```bash
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export FLASH_ATTN_CUDA_ARCHS=80
export MAX_JOBS=4
export NVCC_THREADS=4

python -m pip install packaging ninja
python -m pip install flash-attn==2.7.3 --no-build-isolation --no-cache-dir
```

Why it worked:

- torch was visible during build
- nvcc came from the conda environment (`11.8`)
- the build target was constrained for Ampere GPUs

## 8. CUDA 12 vs CUDA 11.8 mismatch

Local `.bashrc` sets:

```bash
export CUDA_HOME=/usr/local/cuda-12
export LD_LIBRARY_PATH=/usr/local/cuda-12/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-12/bin:$PATH
```

But the repo environment uses:

```text
torch 2.5.1 + cu118
```

This mismatch is harmless for some runtime-only PyTorch use, but it breaks or destabilizes extension builds such as:

- `flash-attn`
- custom CUDA ops
- some `deepspeed` builds

Rule:

- For this repo's `tina` / `tina_eval` environments, prefer the conda environment CUDA toolchain:

```bash
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
```

Recommended verification:

```bash
which nvcc
nvcc --version
python - <<'PY'
import torch
from torch.utils.cpp_extension import CUDA_HOME
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
print(CUDA_HOME)
PY
```

Expected nvcc:

```text
/home/wangls/miniconda3/envs/<env>/bin/nvcc
release 11.8
```

## 9. vLLM installation

Current setup scripts install vLLM from a GitHub release wheel:

```bash
python -m pip install \
  https://github.com/vllm-project/vllm/releases/download/v0.7.2/vllm-0.7.2+cu118-cp38-abi3-manylinux1_x86_64.whl
```

If GitHub access is unreliable, alternatives are:

1. PyPI source build:

```bash
python -m pip install --no-binary=vllm --no-build-isolation vllm==0.7.2
```

2. Direct PyPI wheel:

- only if compatible with the local CUDA / torch stack

Because this repo is pinned around `cu118`, the GitHub `+cu118` wheel remains the most direct match when GitHub is reachable.

## 10. Prefer `python -m pip`

Original setup scripts used bare `pip`.

Risk:

- If the wrong environment is active, `pip` may point to base while `python` points elsewhere.

Fix applied:

- setup scripts now use:

```bash
python -m pip ...
```

This makes the install target match the currently active interpreter.

## 11. Recommended installation order

For both `tina` and `tina_eval`, the reliable order is:

1. Create and activate the conda environment.
2. Install PyTorch stack from conda.
3. Install `xformers`.
4. Install `vllm`.
5. Install the remaining Python dependencies.
6. Install `flash-attn` last with `--no-build-isolation`.

Avoid relying on a single huge `conda env update -f ...` to solve and build everything correctly in one pass.

## 12. Security note

`scripts/set/set_vars.sh` currently contains real-looking W&B and Hugging Face tokens.

These should not live in version-controlled files.

Safer pattern:

```bash
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export HF_TOKEN="${HF_TOKEN:-}"
```

and then provide secrets from the shell or a local untracked file.

## 13. Practical recovery checklist

When the environment starts failing again, check these first:

```bash
conda clean -i -y
env | rg 'proxy|CUDA_HOME'
which python
python -m pip --version
which nvcc
nvcc --version
conda config --show channels
```

If `flash-attn` fails:

- verify torch imports
- verify `CUDA_HOME="$CONDA_PREFIX"`
- reinstall with `--no-build-isolation`

If conda fails with `pkgs/r`:

- inspect `.condarc`
- remove `defaults`
- clear index cache

If GitHub downloads fail:

- verify proxy state
- consider a non-GitHub install path when available
