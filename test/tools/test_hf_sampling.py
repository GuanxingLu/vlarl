"""
test_vllm_sampling.py

Tests if a model with temperature > 0 produces different actions when run multiple times on the same observation.

Usage:
    python experiments/robot/libero/test_vllm_sampling.py \
        --pretrained_checkpoint <CHECKPOINT_PATH> \
        --task_suite_name libero_spatial \
        --temperature 1.0 \
        --num_test_runs 5
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
import time
from libero.libero import benchmark
from ppo.envs.libero_env import VLAEnv
from termcolor import cprint, colored
import wandb
import pprint
import json
import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
import imageio
import cv2

# Append current directory
sys.path.append("../..")
current_path = os.getcwd()
print("Workspace:", current_path)

from run_libero_eval_vllm import GenerateConfig
from ppo.envs.libero_env import VLAEnv
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder, QwenPromptBuilder
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    DATE,
    DATE_TIME,
    get_action,
    get_actions,
    get_actions_batch,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from ppo.utils.util import TimingManager


@dataclass
class SamplingTestConfig(GenerateConfig):
    num_test_runs: int = 5  # Number of times to test the same observation


@draccus.wrap()
def test_vllm_sampling(cfg: SamplingTestConfig) -> None:
    # Set random seed
    set_seed_everywhere(cfg.seed)

    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = cfg.task_suite_name
    
    # Create log directory
    model_pet_name = cfg.load_adapter_checkpoint.split('/')[-1] if cfg.load_adapter_checkpoint else cfg.pretrained_checkpoint.split('/')[-1]
    run_id = f"SAMPLING-TEST-{model_pet_name}-temp-{cfg.temperature}"
    local_log_dir = os.path.join(cfg.local_log_dir, run_id)
    os.makedirs(local_log_dir, exist_ok=True)
    
    # Load model
    model = get_model(cfg)

    # Load prompt builder
    if 'qwen' in cfg.pretrained_checkpoint:
        prompt_builder_fn = QwenPromptBuilder
        cprint(f"Using QwenPromptBuilder for QWEN model", "yellow")
    elif 'v01' in cfg.pretrained_checkpoint:
        prompt_builder_fn = VicunaV15ChatPromptBuilder
    else:
        prompt_builder_fn = PurePromptBuilder

    if cfg.load_adapter_checkpoint is not None:
        # with open(run_dir / "latest_checkpoint_step.txt", "r") as f:
        #     resume_iteration = int(f.read())
        # start_gradient_step_idx = resume_iteration + 1
        # print(f"Resuming training from iteration {resume_iteration} ...")

        dataset_statistics_path = os.path.join(cfg.load_adapter_checkpoint, "dataset_statistics.json")  # HACK: overwrite dataset_statistics
        if os.path.isfile(dataset_statistics_path):
            with open(dataset_statistics_path, "r") as f:
                norm_stats = json.load(f)
            model.norm_stats = norm_stats
            cprint(f"Loaded dataset statistics from {dataset_statistics_path}", "green")

        # Load Model Weights
        model = PeftModel.from_pretrained(model, cfg.load_adapter_checkpoint, is_trainable=False)
        model.merge_and_unload()    # accelerate inference
        model.print_trainable_parameters()
        cprint(f"Loaded adapter weights from {cfg.load_adapter_checkpoint}", "green")

    # [OpenVLA] Check that the model contains the action un-normalization key
    if cfg.model_family == "openvla":
        # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
        # with the suffix "_no_noops" in the dataset name)
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key: {cfg.unnorm_key} not found in VLA `norm_stats`: {model.norm_stats.keys()}! (Please check dataset_statistics.json)"

    # [OpenVLA] Get Hugging Face processor
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    # Initialize environment
    cfg.exp_dir = local_log_dir
    envs = VLAEnv(cfg, mode="eval")
    timer = TimingManager()

    # Get one batch of observations
    obs, _ = envs.reset()
    
    # Store all generated actions
    all_actions = []
    all_response_ids = []
    all_response_logprobs = []
    
    # Run inference multiple times on the same batch
    for run_idx in range(cfg.num_test_runs):
        cprint(f"\nRunning sampling test {run_idx+1}/{cfg.num_test_runs}", "cyan")
        
        generation_start_time = time.time()

        pixel_values = obs["pixel_values"]
        prompts = obs["prompts"]

        generation_start_time = time.time()
        actions, action_tokens, action_log_prob = get_actions(cfg, model, obs=pixel_values, task_label=prompts, 
                                                              pre_thought=None, processor=processor, prompt_builder_fn=prompt_builder_fn,
                                                              temperature=cfg.temperature)
        # actions, _, _ = get_actions_batch(cfg, model, obs=pixel_values, task_label=prompts, pre_thought=None, processor=processor, prompt_builder_fn=prompt_builder_fn)
        print(f"🔥🔥🔥 Action generation time: {time.time() - generation_start_time:.2f} seconds")
        print(f"Run {run_idx+1} generation time: {time.time() - generation_start_time:.2f} seconds")
        
        # Save the actions for comparison
        all_actions.append(actions)
        # all_response_ids.append(action_tokens)
        # all_response_logprobs.append(action_log_prob)
        
        # Print the first action from each run for comparison
        if len(actions) > 0:
            cprint(f"Sample action from run {run_idx+1}: {actions[0]}", "yellow")
            # cprint(f"Sample action_tokens from run {run_idx+1}: {action_tokens[0]}", "yellow")
            # cprint(f"Sample action_log_prob from run {run_idx+1}: {action_log_prob[0]}", "yellow")
    
    # Compare actions to check for sampling variation
    if cfg.num_test_runs > 1:
        cprint("\n--- Sampling Comparison Results ---", "green")
        
        # For each environment in the batch
        for env_idx in range(len(all_actions[0])):
            actions_for_env = [run_actions[env_idx] for run_actions in all_actions]
            
            # Check if actions are identical across runs
            all_identical = all(np.array_equal(actions_for_env[0], action) for action in actions_for_env[1:])
            
            if all_identical:
                cprint(f"Environment {env_idx+1}: All {cfg.num_test_runs} runs produced IDENTICAL actions", "red")
            else:
                cprint(f"Environment {env_idx+1}: Actions varied across runs - sampling is working", "green")
                
                # Calculate variance to quantify the differences
                actions_array = np.array(actions_for_env)
                variance = np.var(actions_array, axis=0)
                mean_variance = np.mean(variance)
                cprint(f"  Mean action variance: {mean_variance:.6f}", "cyan")
    
    # Cleanup
    envs.close()
    timer.close()
    
    cprint("\nSampling test complete!", "green")
    
    # Summary message
    if cfg.temperature > 0.0:
        cprint(f"Temperature was set to {cfg.temperature} - sampling should be observed", "yellow")
    else:
        cprint(f"Temperature was set to {cfg.temperature} - actions should be deterministic", "yellow")


if __name__ == "__main__":
    test_vllm_sampling()