#!/bin/bash
#SBATCH -J test_libero
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH -t 2:00:00
#SBATCH -o logs/test_libero_%j.out
#SBATCH -e logs/test_libero_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate py310

# EGL_PLATFORM=headless
BACKEND=egl
# BACKEND=glx
# BACKEND=osmesa
export MESA_GL_VERSION_OVERRIDE=4.1
export PYOPENGL_PLATFORM=${BACKEND}
export MUJOCO_GL=${BACKEND}
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LIBGL_ALWAYS_SOFTWARE=1

# NOTE: set LIBERO_REPO_PATH to the path of the LIBERO repo
export LIBERO_REPO_PATH=~/guanxing/LIBERO
export LIBERO_CONFIG_PATH=${LIBERO_REPO_PATH}
export PYTHONPATH=${LIBERO_REPO_PATH}:$PYTHONPATH

mkdir -p logs
NOW=$(date +%Y%m%d_%H%M%S)

python -u tests/test_libero.py \
    2>&1 | tee logs/test_libero_${NOW}.log
