#!/bin/bash
# Usage: 
#   bash scripts/train_rl_vllm_ray_fsdp.sh <gpus> <task_ids>
# Example: 
#   bash scripts/train_rl_vllm_ray_fsdp.sh 2,3,4,5,6,7 0,1,2,3,4,5,6,7,8,9
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
# ================================

# export NCCL_P2P_DISABLE=1
# export NCCL_BUFFSIZE=67108864   # 64MiB, default is 4MiB
# export RAY_DEDUP_LOGS=0 # log all ray instances
# export VLLM_LOGGING_LEVEL=DEBUG

export MESA_GL_VERSION_OVERRIDE=4.1
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl
# export MUJOCO_GL=glx
# export MUJOCO_GL=glfw
# export MUJOCO_GL=osmesa

# data
# POSTFIX=spatial
# POSTFIX=goal
# POSTFIX=object
POSTFIX=10
DATA_NAME=libero_${POSTFIX}
DATA_ROOT=${DATA_NAME}_no_noops

# Total H20 GPUs (lora)
per_device_train_batch_size=16
local_rollout_batch_size=10

# Total H20 GPUs (full)
# per_device_train_batch_size=4
# local_rollout_batch_size=10

# Total 2 A100 GPUs
# per_device_train_batch_size=16
# local_rollout_batch_size=10

# Total 8 3090 GPUs
# per_device_train_batch_size=1
# local_rollout_batch_size=1

# GPU allocation
GPUS=${1:-"0,1,2,3"}
MASTER_ADDR=localhost
MASTER_PORT=12345
NUM_GPUS=$(echo $GPUS | tr ',' '\n' | wc -l)
ACTOR_GPUS=$((NUM_GPUS - 1))    # the last GPU is used for vllm
TOTAL_TASKS=$((ACTOR_GPUS * local_rollout_batch_size))
# TASK_IDS=${2:-$(printf "0,%.0s" $(seq 1 $((TOTAL_TASKS))))} # Repeat 0 TOTAL_TASKS-1 times
# TASK_IDS=${TASK_IDS%,} # Remove tailing comma

echo "GPUS=${GPUS}"
echo "TASK_SUITE_NAME=${DATA_NAME}"
echo "TOTAL_TASKS=${TOTAL_TASKS}"
# echo "TASK_IDS=${TASK_IDS}"
echo "ACTOR_GPUS=${ACTOR_GPUS}"
echo "per_device_train_batch_size=${per_device_train_batch_size}"
echo "local_rollout_batch_size=${local_rollout_batch_size}"

# --pretrained_checkpoint "MODEL/openvla-7b-finetuned-libero-${POSTFIX}" \
# --pretrained_checkpoint "MODEL/openvla-7b" \

# CUDA_VISIBLE_DEVICES=$GPUS python \
CUDA_VISIBLE_DEVICES=$GPUS /opt/conda/envs/vlarl/bin/python \
    ppo_vllm_ray_fsdp_v3.py \
    --pretrained_checkpoint "MODEL/openvla-7b-finetuned-libero-${POSTFIX}" \
    --data_root_dir ./data/modified_libero_rlds \
    --dataset_name ${DATA_ROOT} \
    --task_suite_name ${DATA_NAME} \
    --num_trials_per_task 50 \
    --run_root_dir "checkpoints/${DATA_ROOT}/root" \
    --adapter_tmp_dir "checkpoints/${DATA_ROOT}/adapter" \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --local_mini_batch_size ${per_device_train_batch_size} \
    --local_rollout_batch_size ${local_rollout_batch_size} \
    --local_rollout_forward_batch_size ${local_rollout_batch_size} \
    --actor_num_gpus_per_node "[${ACTOR_GPUS}]" \
    --temperature 1.6 \
    --num_epochs 1 \
    --value_init_steps 2 \
    --learning_rate 5e-6 \
    --value_learning_rate 5e-5 \
    --policy_max_grad_norm 1.0 \
    --value_max_grad_norm 1.0 \
    --cliprange_high 0.4 \
    --cliprange_low 0.2 \
    --gamma 1.0 \
    --num_steps 512 \
    --max_env_length 512 \
    --total_episodes 100000 \
    --vllm_tensor_parallel_size 1 \
    --vllm_enforce_eager True \
    --enable_prefix_caching False \
    --gpu_memory_utilization 0.9 \
    --use_lora True \
    --enable_gradient_checkpointing False \
    --sharding_strategy "shard-grad-op" \
    --offload False \
    --use_value_model True \
    --value_model_type "vla" \
    --value_use_lora False \
    --clip_vloss False \
    --norm_adv False \
    --use_curriculum True \
    --curriculum_temp 1.0 \
    --success_history_window 20 \
    --curriculum_recompute_freq 10 \
    --save_freq 20 \
    --eval_freq 20 \
    --save_video True \
    --use_wandb False \
    --wandb_offline False \
    --wandb_project openvla \
    --wandb_entity openvla_cvpr \
    --debug False

# --task_ids "[${TASK_IDS}]" \

# goal:
# --max_env_length 512 \