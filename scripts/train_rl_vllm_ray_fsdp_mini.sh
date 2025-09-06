#!/bin/bash
# Usage: bash scripts/train_rl_vllm_ray_fsdp_mini.sh <gpus> <task_ids>
# Example: bash scripts/train_rl_vllm_ray_fsdp_mini.sh 2,3,4,5,6,7 0,1,2,3,4,5,6,7,8,9
# ================================

# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1
# export NCCL_BUFFSIZE=67108864   # 64MiB, default is 4MiB
export RAY_DEDUP_LOGS=0 # log all ray instances
export MESA_GL_VERSION_OVERRIDE=4.1
export PYOPENGL_PLATFORM=egl

# data
# POSTFIX=spatial
POSTFIX=goal
# POSTFIX=object
# POSTFIX=10
DATA_NAME=libero_${POSTFIX}
DATA_ROOT=${DATA_NAME}_no_noops
per_device_train_batch_size=1
local_rollout_batch_size=4

# GPU allocation
GPUS=${1:-"0,1,2,3"}
MASTER_ADDR=localhost
MASTER_PORT=12345
NUM_GPUS=$(echo $GPUS | tr ',' '\n' | wc -l)
ACTOR_GPUS=$((NUM_GPUS - 1))    # the last GPU is used for vllm
TOTAL_TASKS=$((ACTOR_GPUS * local_rollout_batch_size))
TASK_IDS=${2:-$(printf "0,%.0s" $(seq 1 $((TOTAL_TASKS))))} # Repeat 0 TOTAL_TASKS-1 times
TASK_IDS=${TASK_IDS%,} # Remove trailing comma

echo "GPUS=${GPUS}"
echo "TASK_SUITE_NAME=${DATA_NAME}"
echo "TOTAL_TASKS=${TOTAL_TASKS}"
echo "TASK_IDS=${TASK_IDS}"
echo "ACTOR_GPUS=${ACTOR_GPUS}"
echo "per_device_train_batch_size=${per_device_train_batch_size}"
echo "local_rollout_batch_size=${local_rollout_batch_size}"

CUDA_VISIBLE_DEVICES=$GPUS python \
    ppo_vllm_ray_fsdp_v3.py \
    --pretrained_checkpoint "MODEL/openvla-7b-finetuned-libero-${POSTFIX}" \
    --data_root_dir ./data/modified_libero_rlds \
    --dataset_name ${DATA_ROOT} \
    --task_suite_name ${DATA_NAME} \
    --num_trials_per_task 50 \
    --run_root_dir "checkpoints/debug/root" \
    --adapter_tmp_dir "checkpoints/debug/adapter" \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --local_mini_batch_size ${per_device_train_batch_size} \
    --local_rollout_batch_size ${local_rollout_batch_size} \
    --local_rollout_forward_batch_size ${local_rollout_batch_size} \
    --actor_num_gpus_per_node "[${ACTOR_GPUS}]" \
    --task_ids "[${TASK_IDS}]" \
    --temperature 1.5 \
    --num_epochs 1 \
    --value_init_steps 0 \
    --learning_rate 4e-5 \
    --value_learning_rate 4e-5 \
    --policy_max_grad_norm 1.0 \
    --value_max_grad_norm 5.0 \
    --num_steps 4 \
    --max_env_length 400 \
    --total_episodes 100000 \
    --vllm_tensor_parallel_size 1 \
    --vllm_enforce_eager True \
    --enable_prefix_caching False \
    --gpu_memory_utilization 0.9 \
    --use_lora True \
    --enable_gradient_checkpointing False \
    --sharding_strategy "full-shard" \
    --offload False \
    --use_value_model False \
    --value_model_type "vla" \
    --value_use_lora False \
    --clip_vloss False \
    --norm_adv False \
    --use_curriculum True \
    --curriculum_temp 0.5 \
    --curriculum_min_prob 0.05 \
    --success_history_window 10 \
    --curriculum_recompute_freq 10 \
    --save_freq 100000 \
    --save_video True \
    --use_wandb False \
    --wandb_offline False \
    --wandb_project openvla \
    --wandb_entity openvla_cvpr \
    --debug True
