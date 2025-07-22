# ERROR CATCH

- >>> from flash_attn import flash_attn_2_cuda
ImportError: flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so: undefined symbol: _ZN3c104cuda9SetDeviceEi

Solution:
```bash
pip uninstall flash-attn
pip install "flash-attn==2.5.8" --no-build-isolation
```

- [Tensorflow] Cannot dlopen some GPU libraries

Solution: Reinstall tensorflow
```bash
pip uninstall tensorflow
pip install tensorflow  # will install tensorflow 2.18.0
```

Conduct `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"` to check.

- [Tensorflow] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence

Ref: https://github.com/tensorflow/tensorflow/issues/62963
This doesn't affect the program.

- [vllm] assert "factor" in rope_scaling

Ref: https://github.com/vllm-project/vllm/issues/8388
Upgrade transformers to:
```bash
pip install --upgrade git+https://github.com/huggingface/transformers.git
```

- Evaluate in a headless machine:

Ref:
- https://pytorch.org/rl/main/reference/generated/knowledge_base/MUJOCO_INSTALLATION.html
- https://github.com/openvla/openvla/issues/108
```bash
# If you have sudo:
sudo apt-get install libglfw3 libglew2.0 libgl1-mesa-glx libosmesa6

# Otherwise:
conda install -c conda-forge glew
conda install -c conda-forge mesalib
conda install -c anaconda mesa-libgl-cos6-x86_64
conda install -c menpo glfw3

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# test
python test/test_libero.py
```

- vllm error: Fatal Python error: Floating point exception
Ref: https://github.com/vllm-project/vllm/issues/4392

```bash
pip install nvidia-cublas-cu12==12.4.5.8
```