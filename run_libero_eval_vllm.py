"""
run_libero_eval.py

Runs a model in a LIBERO simulation environment.

Usage:
    # OpenVLA:
    # IMPORTANT: Set `center_crop=True` if model is fine-tuned with augmentations
    python experiments/robot/libero/run_libero_eval.py \
        --model_family openvla \
        --pretrained_checkpoint <CHECKPOINT_PATH> \
        --task_suite_name [ libero_spatial | libero_object | libero_goal | libero_10 | libero_90 ] \
        --center_crop [ True | False ] \
        --run_id_note <OPTIONAL TAG TO INSERT INTO RUN ID FOR LOGGING> \
        --use_wandb [ True | False ] \
        --wandb_project <PROJECT> \
        --wandb_entity <ENTITY>
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Callable, Iterator, List, Literal, Optional, Tuple, Union
import draccus
import numpy as np
import tqdm
from libero.libero import benchmark
from termcolor import cprint, colored
import wandb
import pprint

from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder, QwenPromptBuilder
from ppo.envs.libero_env import VLAEnv
from ppo.utils.vllm_utils2 import create_vllm_engines
from vllm import SamplingParams
import time
import ray
import threading

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
current_path = os.getcwd()
print("Workspace:", current_path)

from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    set_seed_everywhere,
)
from ppo.utils.util import TimingManager


@dataclass
class GenerateConfig:
    # fmt: off
    vla_path: str = "openvla-7b"       # OpenVLA model path
    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    # Environment Parameters
    task_suite_name: str = "libero_spatial"
    """Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90"""
    num_steps_wait: int = 10
    """Number of steps to wait for objects to stabilize in sim"""
    num_tasks_per_suite: int = 10
    """Number of tasks per suite"""
    num_trials_per_task: int = 50
    """Number of rollouts per task"""
    n_rollout_threads: int = 10
    """Number of parallel vec environments"""
    task_ids: Optional[List[int]] = None
    """Task ids to run"""
    max_env_length: int = 0
    """0 for default libero length"""
    env_gpu_id: int = 0
    """GPU id for the vectorized environments"""
    query_length: int = 64
    """Length of the query"""
    save_video: bool = False
    """Whether to save videos"""
    penalty_reward_value: float = 0.0
    """Penalty reward value"""
    non_stop_penalty: bool = False
    """Whether to penalize responses that do not contain `stop_token_id`"""
    verify_reward_value: float = 1.0

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)
    return_thought: bool = False                     # whether return decoded thought chain
    verbose: bool = False                            # Verbose mode for debugging
    subgoal_steps: int = 5                           # Number of steps to take for each subgoal (keep thought chain consistent)
    # fmt: on

    load_adapter_checkpoint: Optional[str] = None    # Path to adapter checkpoint to load
    save_images: bool = False                        # Whether to save images besides videos
    
    # generation config
    response_length: int = 8
    """the length of the response"""
    stop_token_id: Optional[int] = None
    """the truncation token id"""
    min_response_length: int = 0
    """stop only after this many tokens"""
    temperature: float = 1.0
    """the sampling temperature"""
    verify_reward_value: float = 10.0
    """the reward value for responses that do not contain `stop_token_id`"""
    penalty_reward_value: float = -1.0
    """the reward value for responses that do not contain `stop_token_id`"""
    non_stop_penalty: bool = False
    """whether to penalize responses that do not contain `stop_token_id`"""
    number_envs_per_task: int = 1
    """the number of samples to generate per prompt, useful for easy-star"""

    # ray
    vllm_num_engines: int = 1
    """number of vLLM Engines, set to 0 to disable vLLM"""
    vllm_tensor_parallel_size: int = 1
    """tensor parallel size of vLLM Engine for multi-GPU inference"""
    vllm_enforce_eager: bool = False
    """whether to enforce eager mode for vLLM -- slow inference but needed for multi-node"""
    vllm_sync_backend: str = "nccl"
    """DeepSpeed -> vLLM weight sync backend"""
    enable_prefix_caching: bool = False
    """whether to enable prefix caching"""
    gpu_memory_utilization: float = 0.9
    """pre-allocated GPU memory utilization for vLLM"""


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = cfg.task_suite_name

    # Load model
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
        # enable_lora=True if cfg.load_adapter_checkpoint is not None else False,
        # norm_stats=norm_stats,
    )
    generation_config = SamplingParams(
        temperature=cfg.temperature,    # for greedy sampling
        max_tokens=cfg.response_length,
        include_stop_str_in_output=False,
        detokenize=False,
        n=1,
        seed=cfg.seed,
        logprobs=1,
    )
    print(f"generation_config: {generation_config}")

    # (Optional) Load processor
    # processor = get_processor(cfg)
    processor = None

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
                if processor is None:
                    prompts = g_queries_list["prompts"]
                    prompts = ["<PAD>" + prompt + "▁" for prompt in prompts]
                    # print(f"🔥🔥🔥 prompts: {prompts}")
                    llm_inputs = [
                        {
                            "prompt": prompt,
                            "multi_modal_data": {"image": pixel_value},
                        } for prompt, pixel_value in zip(prompts, pixel_values)
                    ]
                else:
                    prompts = g_queries_list["prompts"]
                    prompts = ["<PAD>" + prompt + "▁" for prompt in prompts]
                    prompt_token_ids = processor.tokenizer(prompts, return_tensors="pt").input_ids
                    print(f"🔥🔥🔥 prompt_token_ids: {prompt_token_ids}")
                    llm_inputs = [
                        {
                            "prompt_token_ids": prompt_token_id,
                            "multi_modal_data": {"image": pixel_value},
                        } for prompt_token_id, pixel_value in zip(prompt_token_ids, pixel_values)
                    ]

                # generation_start_time = time.time()
                actions, response_ids, response_logprobs = ray.get(
                    llm.predict_action.remote(
                        llm_inputs,
                        sampling_params=generation_config, 
                        use_tqdm=False,
                        unnorm_key=cfg.unnorm_key,
                        # lora_request=LoRARequest("vla_adapter", 1, cfg.load_adapter_checkpoint) if cfg.load_adapter_checkpoint is not None else None
                    )
                )
                # print(f"{response_logprobs=}")
                print(f"🔥🔥🔥 Action generation time: {time.time() - generation_start_time:.2f} seconds")
                response_ids_Q.put(actions)

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
    print("vllm generate thread starts")

    # Load prompt builder
    if 'qwen' in cfg.pretrained_checkpoint:
        prompt_builder_fn = QwenPromptBuilder
        cprint(f"Using QwenPromptBuilder for QWEN model", "yellow")
    elif 'v01' in cfg.pretrained_checkpoint:
        prompt_builder_fn = VicunaV15ChatPromptBuilder
    else:
        prompt_builder_fn = PurePromptBuilder

    # Initialize local logging
    # max_pet_name_len = len("openvla-7b")
    # model_pet_name = cfg.pretrained_checkpoint.split('/')[-1][:max_pet_name_len]
    model_pet_name = cfg.load_adapter_checkpoint.split('/')[-1] if cfg.load_adapter_checkpoint else cfg.pretrained_checkpoint.split('/')[-1]
    run_id = f"EVAL-{cfg.task_suite_name}-{model_pet_name}-t-{cfg.temperature}-s-{cfg.seed}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    local_log_dir = os.path.join(cfg.local_log_dir, run_id)
    os.makedirs(local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    cprint(f"Logging to local log file: {local_log_filepath}", "cyan")

    # Initialize Weights & Biases logging as well
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    # cfg.exp_dir = cfg.pretrained_checkpoint if cfg.load_adapter_checkpoint is None else cfg.load_adapter_checkpoint    # for saved video
    cfg.exp_dir = local_log_dir
    # Clear exp_dir to avoid video duplication
    video_dir = os.path.join(cfg.exp_dir, "rollouts")
    cprint(f"Clearing existing videos in {video_dir}", "red")
    if os.path.exists(video_dir):
        for f in os.listdir(video_dir):
            if f.endswith(".mp4"):
                os.remove(os.path.join(video_dir, f))

    # Initialize vectorized environment
    envs = VLAEnv(cfg, mode="eval")
    timer = TimingManager()

    # Main evaluation loop
    total_episodes = 0
    total_successes = 0
    pre_thought = None
    
    # Reset environments
    obs, infos = envs.reset()
        
    while True:
        # Batch inference
        # with torch.no_grad(), timer.timer("model_inference"):
        #     pixel_values = obs["pixel_values"]
        #     prompts = obs["prompts"]
        #     actions, _, _ = get_actions(cfg, model, obs=pixel_values, task_label=prompts, pre_thought=None, processor=processor, prompt_builder_fn=prompt_builder_fn)
        
        with timer.timer("model_inference"):
            generation_start_time = time.time()
            param_prompt_Q.put(obs)
            actions = response_ids_Q.get()
            print(f"🔥🔥🔥 Action generation time: {time.time() - generation_start_time:.2f} seconds")
        
        # Step environments
        with timer.timer("env_step"):
            next_obs, rewards, dones, infos = envs.step(actions)
        
        # Update success count
        total_successes += sum([r > 0 for r in rewards])
        
        # Break if all episodes in batch are done
        if np.allclose(envs.initial_state_ids, -1):
        # if next_obs is None:
            break
            
        obs = next_obs
        
    total_episodes = cfg.num_trials_per_task * cfg.num_tasks_per_suite
    
    # Log progress
    success_rate = total_successes / total_episodes
    print(f"Episodes completed: {total_episodes}")
    print(f"Success rate: {success_rate:.2%}")
    log_file.write(f"Episodes completed: {total_episodes}\n")
    log_file.write(f"Success rate: {success_rate:.2%}\n")
    
    # Push total metrics and local log file to wandb
    time_infos = timer.get_log()
    log_infos = {
        "success_rate/total": float(total_successes) / float(total_episodes),
        "num_episodes/total": total_episodes,
    }
    log_infos.update(time_infos)

    for k, v in log_infos.items():
        log_file.write(f"{k}: {v}\n")
    pprint.pprint(log_infos)

    if cfg.use_wandb:
        wandb.log(log_infos)
        wandb.save(local_log_filepath)

    # Cleanup
    log_file.close()
    timer.close()
    envs.close()
    
    # Cleanup vLLM thread
    param_prompt_Q.put(None)  # Signal thread to stop
    thread.join()  # Wait for thread to finish
    
    ray.shutdown()

    cprint("Evaluation complete!", "green")


if __name__ == "__main__":
    eval_libero()
