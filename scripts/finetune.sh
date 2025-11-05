# !/bin/bash

DATA_ROOT=libero_spatial_no_noops

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nnodes 1 --nproc-per-node 4 finetune.py \
  --vla_path ./openvla-7b \
  --data_root_dir ./data/real_world_ntu_rlds \
  --dataset_name ${DATA_ROOT} \
  --run_root_dir checkpoints/${DATA_ROOT}/root \
  --adapter_tmp_dir checkpoints/${DATA_ROOT}/adapter \
  --lora_rank 32 \
  --batch_size 8 \
  --grad_accumulation_steps 4 \
  --learning_rate 5e-4 \
  --image_aug True \
  --wandb_project openvla \
  --wandb_entity  openvla_cvpr \
  --save_steps 10000


  # --vla_path ./MODEL/prism-qwen25-extra-dinosiglip-224px-0_5b-hf \
  # --data_root_dir ./data/modified_libero_rlds \