#!/bin/bash

# MODEL_PATH=ppo+libero_spatial_no_noops+trials50+ns128+maxs128+rb10+tb16+lr-5e-06+vlr-0.0001+s-1+lora
# MODEL_PATH=ppo+libero_spatial_no_noops+tasks1+trials50+ns128+maxs150+rb10+tb16+lr-1e-05+vlr-0.0001+s-1+lora
# MODEL_PATH=ppo+libero_goal_no_noops+rb10+tb16+lr-5e-06+vlr-0.0001+s-1+lora
# MODEL_PATH=ppo+libero_goal_no_noops+rb10+tb16+lr-2e-05+vlr-0.0005+s-1+lora
MODEL_PATH=ppo+libero_10_no_noops+rb10+tb16+lr-5e-06+vlr-0.0001+s-1+lora

SERVER_IP=195.26.233.55
SERVER_PORT=41957

echo "Scp from the server"
scp -P ${SERVER_PORT} -r root@${SERVER_IP}:/workspace/vlarl/logs/tensorboard/${MODEL_PATH}/ logs/tensorboard/
# scp -P ${SERVER_PORT} -r root@${SERVER_IP}:/workspace/vlarl/logs/tensorboard/ logs/
# scp -P ${SERVER_PORT} -r root@${SERVER_IP}:/workspace/vlarl/MODEL/openvla-7b-finetuned-libero-10 MODEL/