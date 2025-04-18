#!/bin/bash

# POSTFIX=spatial
POSTFIX=goal
# POSTFIX=10
DATA_NAME=libero_${POSTFIX}
DATA_ROOT=${DATA_NAME}_no_noops

# 2 GPUs, one for vLLM, one for env
# CUDA_VISIBLE_DEVICES=4,5 python test/test_hf_sampling.py \
CUDA_VISIBLE_DEVICES=4,5 python test/test_vllm_sampling.py \
  --model_family openvla \
  --pretrained_checkpoint "MODEL/openvla-7b-finetuned-libero-${POSTFIX}" \
  --task_suite_name ${DATA_NAME} \
  --num_trials_per_task 50 \
  --num_tasks_per_suite 10 \
  --center_crop True \
  --seed 7 \
  --use_wandb False \
  --wandb_project openvla \
  --wandb_entity  openvla_cvpr \
  --return_thought False \
  --verbose False \
  --save_video True \
  --save_images False \
  --enable_prefix_caching False \
  --vllm_enforce_eager True \
  --gpu_memory_utilization 0.9 \
  --env_gpu_id "1" \
  --temperature 1.0 \
  --num_test_runs 5

# for base
  # --pretrained_checkpoint ./openvla-7b \

# for released checkpoint
  # --pretrained_checkpoint "MODEL/openvla-7b-finetuned-libero-${POSTFIX}" \

# for ours checkpoint
  # --pretrained_checkpoint "checkpoints/libero_goal_no_noops/root/ppo+libero_goal_no_noops+rb10+tb16+lr-2e-05+vlr-0.0005+s-1+lora" \

# for testing
  # --num_trials_per_task 50 \
  # --num_tasks_per_suite 10 \

# for debugging
  # --max_env_length 0 \
  # --num_trials_per_task 2 \
  # --num_tasks_per_suite 2 \

  # --num_trials_per_task 50 \
  # --num_tasks_per_suite 4 \

  # --num_trials_per_task 50 \
  # --num_tasks_per_suite 1 \