# Installation

The following guidance works well for a machine with 3090 GPU | cuda 12.1 | ubuntu 22.04 LTS, a machine with A100 GPU | cuda 12.4 | ubuntu 22.04, and more machines.

For possible errors, please see [ERROR_CATCH.md](ERROR_CATCH.md). If you encounter any other problem, feel free to open an issue.

```bash
# Install ubuntu dependencies
apt-get -y update && apt-get install -y cmake libegl-dev git-lfs

# apt-get install -y libgl1 libx11-dev libglew-dev

# Conda environment setup
conda create -n vlarl python=3.10
conda activate vlarl

# PyTorch installation (CUDA 12.1)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Clone the repo and install it
git clone https://github.com/GuanxingLu/vlarl.git
cd vlarl
pip install -e .
# for LIBERO simulation
pip install -r experiments/robot/libero/libero_requirements.txt

# [vLLM]
cd prismatic/extern/vllm; pip install -e .; cd ../../..

# Install Flash Attention 2 for training (https://github.com/Dao-AILab/flash-attention)
#   =>> If you run into difficulty, try `pip cache remove flash_attn` first
pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
pip install "flash-attn==2.5.5" --no-build-isolation
# NOTE: if the building process is slow, please check https://github.com/mjun0812/flash-attention-prebuild-wheels for prebuilt wheels.

# [Libero]
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e .

# Download SFT checkpoints
mkdir -p MODEL/; cd MODEL/
# e.g., LIBERO-Spatial
git lfs clone https://huggingface.co/openvla/openvla-7b-finetuned-libero-spatial

# Download LIBERO SFT dataset
mkdir -p data/; cd data/
git lfs clone https://huggingface.co/datasets/openvla/modified_libero_rlds

# Set up Weights & Biases for experiment logging
wandb login
```