#!/bin/bash
#SBATCH -J vlarl
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH -t 40:00:00
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

export MESA_GL_VERSION_OVERRIDE=4.1
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl
# NOTE: set LIBERO_REPO_PATH to the path of the LIBERO repo
export LIBERO_REPO_PATH=~/guanxing/LIBERO
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
per_device_train_batch_size=8   # zero2
# per_device_train_batch_size=4   # ddp
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
source ~/miniconda3/etc/profile.d/conda.sh
conda activate py310

# nvidia-smi

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
    --gpu_memory_utilization 0.9 \
    --use_lora True \
    --enable_gradient_checkpointing False \
    --sharding_strategy "full-shard" \
    --offload False \
    --norm_adv False \
    --use_baseline True \
    --baseline_momentum 0.9 \
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
    2>&1 | tee logs/${NOW}.log
