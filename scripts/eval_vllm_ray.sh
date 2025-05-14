#!/bin/bash

POSTFIX=spatial
# POSTFIX=goal
# POSTFIX=object
# POSTFIX=10
DATA_NAME=libero_${POSTFIX}
DATA_ROOT=${DATA_NAME}_no_noops

MODEL_PATH=MODEL/openvla-7b-finetuned-libero-${POSTFIX}
# MODEL_PATH=ppo+libero_spatial_no_noops+trials50+ns128+maxs128+rb10+tb16+lr-5e-06+vlr-0.0001+s-1+lora
# STEP_PATH=step_60

SCP=False
SERVER_IP=195.26.233.41
SERVER_PORT=23341

MERGE=False
EVAL=True

GPUS=${1:-"0,1"}

# 1. scp the model from the server
if [ ${SCP} = True ]; then
  echo "1. Scp the model from the server"
  mkdir -p checkpoints/${DATA_ROOT}/root/${MODEL_PATH}
  scp -P ${SERVER_PORT} -r root@${SERVER_IP}:/workspace/vlarl/checkpoints/${DATA_ROOT}/root/${MODEL_PATH}/${STEP_PATH}/ checkpoints/${DATA_ROOT}/root/${MODEL_PATH}/
fi

# 2. Merge the model with the adapter
if [ ${MERGE} = True ]; then
  echo "2. Merging the model with the adapter"
  CUDA_VISIBLE_DEVICES=$GPUS python vla-scripts/merge.py \
    --vla_path "MODEL/openvla-7b-finetuned-libero-${POSTFIX}" \
    --run_root_dir checkpoints/${DATA_ROOT}/root/${MODEL_PATH} \
    --adapter_tmp_dir checkpoints/${DATA_ROOT}/root/${MODEL_PATH}/${STEP_PATH} \
    --copy_needed_files True
fi

# 3. Evaluate the merged model
# 2 GPUs, one for vLLM, one for env
if [ ${EVAL} = True ]; then
  echo "3. Evaluating the merged model"
  CUDA_VISIBLE_DEVICES=$GPUS python run_libero_eval_vllm.py \
    --model_family openvla \
    --pretrained_checkpoint "MODEL/openvla-7b-finetuned-libero-${POSTFIX}" \
    --local_log_dir debug \
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
    --temperature 1.0
fi

  # --pretrained_checkpoint "MODEL/openvla-7b-finetuned-libero-${POSTFIX}" \
  # --pretrained_checkpoint checkpoints/${DATA_ROOT}/root/${MODEL_PATH} \