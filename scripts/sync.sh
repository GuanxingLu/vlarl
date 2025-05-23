#!/bin/bash

# MODEL_PATH=ppo+libero_spatial_no_noops+trials50+ns128+maxs128+rb10+tb16+lr-5e-06+vlr-0.0001+s-1+lora
# MODEL_PATH=ppo+libero_spatial_no_noops+tasks1+trials50+ns128+maxs150+rb10+tb16+lr-1e-05+vlr-0.0001+s-1+lora
# MODEL_PATH=ppo+libero_goal_no_noops+rb10+tb16+lr-5e-06+vlr-0.0001+s-1+lora
# MODEL_PATH=ppo+libero_goal_no_noops+rb10+tb16+lr-2e-05+vlr-0.0005+s-1+lora
# MODEL_PATH=ppo+libero_10_no_noops+rb10+tb16+lr-5e-06+vlr-0.0001+s-1+lora
MODEL_PATH=ppo+libero_goal_no_noops+tasks10+trials50+ns128+maxs200+rb10+tb16+lr-4e-05+vlr-4e-05+temp-1.5+s-1+lora+cl

# Extract the sub-directory from MODEL_PATH (e.g., libero_goal_no_noops)
# This assumes the desired part is the second component when splitting by '+'
DATA_PATH=$(echo $MODEL_PATH | cut -d'+' -f2)

SOURCE_PATH="/workspace/vlarl/checkpoints/${DATA_PATH}/root/${MODEL_PATH}"

EXCLUDE_PATTERNS=("*/rollouts/")

SERVER_IP=195.26.233.9
SERVER_PORT=49352

echo "Scp from the server" # This can be updated to "Syncing from server" if all scp are replaced

# Construct rsync exclude options
declare -a rsync_exclude_opts_array=()
if [ ${#EXCLUDE_PATTERNS[@]} -gt 0 ]; then
  for pattern in "${EXCLUDE_PATTERNS[@]}"; do
    rsync_exclude_opts_array+=(--exclude="${pattern}")
  done
fi

# echo the command to do
echo "rsync -avz --progress -e \"ssh -p ${SERVER_PORT}\" ${rsync_exclude_opts_array[@]} root@${SERVER_IP}:${SOURCE_PATH} checkpoints/${DATA_PATH}/"

rsync -avz --progress -e "ssh -p ${SERVER_PORT}" "${rsync_exclude_opts_array[@]}" root@${SERVER_IP}:${SOURCE_PATH} checkpoints/${DATA_PATH}/
# rsync -avz --progress -e "ssh -p ${SERVER_PORT}" "${rsync_exclude_opts_array[@]}" root@${SERVER_IP}:/workspace/vlarl/logs/tensorboard/${MODEL_PATH}/ logs/tensorboard/
