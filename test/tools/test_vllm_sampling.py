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
from queue import Queue
import threading
import time
from typing import List, Optional, Union

import draccus
import numpy as np
import ray
from termcolor import cprint
from vllm import SamplingParams
import torch

# Append current directory
sys.path.append("../..")
current_path = os.getcwd()
print("Workspace:", current_path)

from run_libero_eval_vllm import GenerateConfig
from ppo.envs.libero_env import VLAEnv
from ppo.utils.vllm_utils2 import create_vllm_engines
from experiments.robot.robot_utils import set_seed_everywhere
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
    
    # Load vLLM engines
    max_len = 256 + 50
    vllm_engines = create_vllm_engines(
        cfg.vllm_num_engines,
        cfg.vllm_tensor_parallel_size,
        cfg.vllm_enforce_eager,
        cfg.pretrained_checkpoint,
        revision=None,
        seed=cfg.seed,
        enable_prefix_caching=cfg.enable_prefix_caching,
        max_model_len=max_len,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
    )
    
    generation_config = SamplingParams(
        temperature=cfg.temperature,
        max_tokens=cfg.response_length,
        include_stop_str_in_output=False,
        detokenize=False,
        n=1,
        top_p=1.0,
        # seed=cfg.seed,
        logprobs=1,
    )
    print(f"Generation config: {generation_config}")

    def vllm_generate(
            generation_config: SamplingParams,
            response_ids_Q: Queue,
            param_prompt_Q: Queue,
        ):
            llm = vllm_engines[0]
            while True:
                g_queries_list = param_prompt_Q.get()
                if g_queries_list is None:
                    break

                pixel_values = g_queries_list["pixel_values"]
                prompts = g_queries_list["prompts"]
                prompts = ["<PAD>" + prompt + "▁" for prompt in prompts]
                print(f"🔥🔥🔥 prompts: {prompts}")
                
                llm_inputs = [
                    {
                        "prompt": prompt,
                        "multi_modal_data": {"image": pixel_value},
                    } for prompt, pixel_value in zip(prompts, pixel_values)
                ]

                generation_start_time = time.time()
                actions, response_ids, response_logprobs = ray.get(
                    llm.predict_action.remote(
                        llm_inputs,
                        sampling_params=generation_config, 
                        use_tqdm=False,
                        unnorm_key=cfg.unnorm_key,
                    )
                )
                print(f"🔥🔥🔥 Action generation time: {time.time() - generation_start_time:.2f} seconds")
                response_ids_Q.put((actions, response_ids, response_logprobs))

    # Setup communication queues
    response_ids_Q = Queue(maxsize=1)
    param_prompt_Q = Queue(maxsize=1)
    thread = threading.Thread(
                target=vllm_generate,
                args=(
                    generation_config,
                    response_ids_Q,
                    param_prompt_Q,
                ),
            )
    thread.start()
    print("vLLM generate thread started")

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
        param_prompt_Q.put(obs)
        actions, response_ids, response_logprobs = response_ids_Q.get()
        print(f"Run {run_idx+1} generation time: {time.time() - generation_start_time:.2f} seconds")
        
        # Save the actions for comparison
        all_actions.append(actions)
        all_response_ids.append(response_ids)
        all_response_logprobs.append(response_logprobs)
        
        # Print the first action from each run for comparison
        sample_id = 3
        if len(actions) > sample_id:
            cprint(f"Sample action from run {run_idx+1}: {actions[sample_id]}", "yellow")
            cprint(f"Sample response_ids from run {run_idx+1}: {response_ids[sample_id]}", "yellow")
            cprint(f"Sample response_logprobs from run {run_idx+1}: {response_logprobs[sample_id]}", "yellow")
    
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
    
    # Cleanup vLLM thread
    param_prompt_Q.put(None)  # Signal thread to stop
    thread.join()  # Wait for thread to finish
    
    ray.shutdown()
    
    cprint("\nSampling test complete!", "green")
    
    # Summary message
    if cfg.temperature > 0.0:
        cprint(f"Temperature was set to {cfg.temperature} - sampling should be observed", "yellow")
    else:
        cprint(f"Temperature was set to {cfg.temperature} - actions should be deterministic", "yellow")


if __name__ == "__main__":
    test_vllm_sampling()