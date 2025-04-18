# !/bin/bash

CUDA_VISIBEL_DEVICES=5 python vla-scripts/extern/convert_minivla_weights_to_hf.py \
    --openvla_model_path_or_id MODEL/prism-qwen25-extra-dinosiglip-224px-0_5b/ \
    --output_hf_model_local_path MODEL/prism-qwen25-extra-dinosiglip-224px-0_5b-hf2
