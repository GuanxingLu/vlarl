#!/bin/bash
# Usage: 
#   bash scripts/train_rl_vllm_ray_fsdp.sh <gpus> <task_ids>
# Example: 
#   bash scripts/train_rl_vllm_ray_fsdp.sh
#   bash scripts/train_rl_vllm_ray_fsdp.sh 0,1,2,3,4,5,6,7 0,1,2,3,4,5,6,7,8,9
# Devices: more than 2 A100 GPUs; 6 GPUs for RTX 3090s backward, but the second broadcast will oom
# Parameters:
# Rollout phase: num_envs = local_rollout_batch_size * world_size
# e.g. 2 GPUs, local_rollout_batch_size = 1, num_envs = 1 * 2 = 2
# Training phase: num_mini_batches = local_rollout_batch_size * num_steps / local_mini_batch_size
# e.g. 2 GPUs, local_rollout_batch_size = 1, num_steps = 128, local_mini_batch_size = 8, num_mini_batches = 1 * 128 / 8 = 16
# Curriculum:
# task num = 10, initial state num = 50 -> average curriculum prob = 0.002
# Expectation:
# 10 tasks / 10 local_rollout_batch_size x 50 initial states x 200 steps x 2.5s = 25000s = ~7 hours


# A GRPO example:
# GPU0: group_0 = (task_0, task_0, ..), group_1 = (task_1, task_1, ..) (local_rollout_batch_size, e.g., 8)
# GPU1: group_0 = (task_2, task_2, ..), group_1 = (task_3, task_3, ..)
# ...
# ================================

# export NCCL_P2P_DISABLE=1
# export NCCL_BUFFSIZE=67108864   # 64MiB, default is 4MiB
# export RAY_DEDUP_LOGS=0 # log all ray instances
# export VLLM_LOGGING_LEVEL=DEBUG

export MESA_GL_VERSION_OVERRIDE=4.1
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LIBGL_ALWAYS_SOFTWARE=1
# NOTE: set LIBERO_REPO_PATH to the path of the LIBERO repo
export LIBERO_REPO_PATH=~/LIBERO
export LIBERO_CONFIG_PATH=${LIBERO_REPO_PATH}
export PYTHONPATH=${LIBERO_REPO_PATH}:$PYTHONPATH

export WANDB_API_KEY=fab158aa77d0a22d98d5ca0656237d0541c1bad0

# data
POSTFIX=spatial
# POSTFIX=goal
# POSTFIX=object
# POSTFIX=10
DATA_NAME=libero_${POSTFIX}
DATA_ROOT=${DATA_NAME}_no_noops

# Total H20 GPUs (lora)
# per_device_train_batch_size=16

# Total H20 GPUs (full)
# per_device_train_batch_size=8   # zero2
per_device_train_batch_size=4   # ddp
local_num_groups=1
local_group_size=8

# GPU allocation
GPUS=0,1
MASTER_ADDR=localhost
MASTER_PORT=12345
NUM_GPUS=$(echo $GPUS | tr ',' '\n' | wc -l)
ACTOR_GPUS=$((NUM_GPUS - 1))    # the last GPU is used for vllm

TASK_IDS="1"

mkdir -p logs
NOW=$(date +%Y%m%d_%H%M%S)

CUDA_VISIBLE_DEVICES=$GPUS python \
    grpo_vllm_ray_fsdp.py \
    --pretrained_checkpoint "MODEL/openvla-7b-finetuned-libero-${POSTFIX}" \
    --data_root_dir ./data/modified_libero_rlds \
    --dataset_name ${DATA_ROOT} \
    --task_suite_name ${DATA_NAME} \
    --num_trials_per_task 50 \
    --eval_num_trials_per_task 5 \
    --task_ids "[${TASK_IDS}]" \
    --run_root_dir "checkpoints/${DATA_ROOT}/root" \
    --adapter_tmp_dir "checkpoints/${DATA_ROOT}/adapter" \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --local_mini_batch_size ${per_device_train_batch_size} \
    --local_num_groups ${local_num_groups} \
    --local_group_size ${local_group_size} \
    --actor_num_gpus_per_node "[${ACTOR_GPUS}]" \
    --temperature 1.7 \
    --num_epochs 1 \
    --learning_rate 8e-6 \
    --policy_max_grad_norm 1.0 \
    --cliprange_high 0.4 \
    --cliprange_low 0.2 \
    --penalty_reward_value -1.0 \
    --num_steps 160 \
    --max_env_length 160 \
    --total_episodes 100000 \
    --vllm_tensor_parallel_size 1 \
    --vllm_enforce_eager True \
    --enable_prefix_caching False \
    --gpu_memory_utilization 0.5 \
    --use_lora True \
    --enable_gradient_checkpointing False \
    --sharding_strategy "full-shard" \
    --offload False \
    --norm_adv True \
    --kl_coef 0.0 \
    --entropy_bonus 0.0 \
    --nonneg_adv False \
    --clip_ploss True \
    --use_curriculum False \
    --curriculum_temp 1.0 \
    --curriculum_min_prob 0.0 \
    --save_freq 100 \
    --eval_freq 100 \
    --init_eval False \
    --save_video True \
    --use_wandb True \
    --wandb_offline False \
    --wandb_project openvla \
    --wandb_entity openvla_cvpr \
    --debug False \
    2>&1 | tee logs/$vlarl_{NOW}.log
