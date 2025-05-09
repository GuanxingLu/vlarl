# Copyright 2025 VLA-RL. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ---------------------------------------------------------------------
# Adapted from
# https://github.com/OpenRLHF/OpenRLHF and https://github.com/allenai/open-instruct
# and https://github.com/vwxyzjn/cleanrl
# which has the following license:
# Copyright 2023 OpenRLHF
# Copyright 2024 AllenAI
# Copyright 2019 CleanRL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from argparse import Namespace
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional
from termcolor import cprint
import draccus
import tqdm
import math
import json
import gc
import time
import random
import shutil
import socket
import threading
import numpy as np
import logging
from queue import Empty, Queue
from typing import Any, Callable, Iterator, List, Literal, Optional, Tuple, Dict, Union
from copy import deepcopy

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision, CPUOffload
from torch.distributed.device_mesh import init_device_mesh
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoModelForVision2Seq, 
    AutoProcessor, 
    AutoConfig, 
    AutoImageProcessor,
    AutoModel,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_scheduler,
    BitsAndBytesConfig,
)
from transformers.processing_utils import ProcessorMixin
from accelerate.utils import is_peft_model
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
import ray
from ray.util.placement_group import PlacementGroup, placement_group
from ray.util.queue import Queue as RayQueue
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from vllm import SamplingParams
from rich.pretty import pprint
from termcolor import cprint
from ppo.envs.libero_env import VLAEnv
from ppo.models.critic import CriticVLA, CriticQwen, CriticFilm
from ppo.models.prm import DummyRM, QwenProcessRM
from ppo.utils.util import TimingManager
from ppo.utils.vllm_utils2 import create_vllm_engines, init_process_group
from ppo.utils.ray_utils import ray_noset_visible_devices, get_physical_gpu_id
from ppo.utils.logging_utils import init_logger
from ppo.envs.base import BaseEnv, EnvOutput
from ppo.utils.fsdp_utils import (
    get_fsdp_wrap_policy_openvla,
    init_fn,
    log_gpu_memory_usage,
)
# OpenVLA-specific imports
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig, VISION_BACKBONE_TO_TIMM_ID
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    disable_dropout_in_model,
    first_true_indices,
    set_seed_everywhere,
    exact_div,
    forward,
    get_reward,
    process_with_padding_side,
    truncate_response,
    add_special_token,
    remove_padding,
    print_rich_single_line_metrics,
)

# to debug ray instance in vscode, ref: https://github.com/ray-project/ray/issues/41953
# import debugpy
# debugpy.listen(("localhost", 5678))
logger = init_logger(__name__)

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ray.init(runtime_env={"env_vars": {"RAY_DEBUG": "legacy"}})

@dataclass 
class Args:
    # Common args
    seed: int = 1
    """Seed of the experiment"""

    # VLA Model args
    vla_path: str = "openvla/openvla-7b"
    """Path to OpenVLA model (on HuggingFace Hub)"""
    load_adapter_checkpoint: Optional[str] = None
    """Path to adapter checkpoint to load"""
    pretrained_checkpoint: str = "openvla/openvla-7b"
    """For data collection"""
    # load_in_8bit: bool = False
    # """(For OpenVLA only) Load with 8-bit quantization"""
    # load_in_4bit: bool = False
    # """(For OpenVLA only) Load with 4-bit quantization"""
    model_family: str = "openvla"
    """Model family"""
    use_fast_tokenizer: bool = False
    """Whether to use fast tokenizer"""
    enable_gradient_checkpointing: bool = False
    """Only save important activations to save memory"""

    # Reward Model args
    prm_model_name_or_path: str = "MODEL/Qwen2-VL-2B-Instruct"
    """Path to reward model (on HuggingFace Hub)"""
    prm_checkpoint_path: Optional[str] = None
    """Path to reward model checkpoint to load"""
    
    # value model
    use_value_model: bool = False
    """whether to use the value model"""
    value_model_path: str = "Qwen2.5-VL-3B-Instruct"
    """the path to the value model"""
    value_model_type: str = "vla"
    """the type of the value model, Options: `vla`, `film` or `qwen`"""
    value_adapter_dir: Optional[str] = None
    """the path to the value model adapter"""
    value_use_lora: bool = False
    """whether to use LoRA for the value model"""
    value_lora_rank: int = 32
    """the rank of the LoRA for the value model"""
    value_lora_dropout: float = 0.0
    """the dropout for the LoRA for the value model"""
    
    # Directory Paths
    data_root_dir: Path = Path("datasets/open-x-embodiment")
    """Path to Open-X dataset directory"""
    dataset_name: str = "droid_wipe"
    """Name of fine-tuning dataset (e.g., `droid_wipe`)"""
    run_root_dir: Path = Path("runs")
    """Path to directory to store logs & checkpoints"""
    adapter_tmp_dir: Path = Path("adapter-tmp")
    """Temporary directory for LoRA weights before fusing"""

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
    context_length: int = 64
    """Length of the query"""

    # for debugging
    verbose: bool = False
    """Whether to print a lot of information"""
    debug: bool = False
    """Whether to run in debug mode (w/o broadcasting to vllm, etc.)"""
    save_video: bool = True
    """Save video of evaluation rollouts"""

    # Dataset
    # shuffle_buffer_size: int = 100_000
    shuffle_buffer_size: int = 1
    """Dataloader shuffle buffer size (can reduce if OOM)"""

    # Augmentation
    image_aug: bool = True
    """Whether to train with image augmentations"""
    center_crop: bool = True
    """Center crop (if trained w/ random crop image aug)"""

    # LoRA Arguments
    use_lora: bool = False
    """Whether to use LoRA fine-tuning"""
    lora_rank: int = 32
    """Rank of LoRA weight matrix"""
    lora_dropout: float = 0.0
    """Dropout applied to LoRA weights"""
    use_quantization: bool = False
    """Whether to 4-bit quantize VLA for LoRA fine-tuning
    => CAUTION: Reduces memory but hurts performance"""
    merge_model: bool = False
    """Whether to merge LoRA weights into model backbone"""
    load_model: bool = False
    """Whether to load model from checkpoint to resume training"""

    # optimizer args
    eps: float = 1e-5
    """The epsilon value for the optimizer"""
    learning_rate: float = 5e-5
    """The initial learning rate for AdamW optimizer."""
    weight_decay: float = 0.0
    """The weight decay for the optimizer"""
    value_learning_rate: float = 5e-5
    """The initial learning rate for AdamW optimizer."""
    policy_lr_scheduler_type: str = "constant"
    """Which scheduler to use"""
    policy_warm_up_steps: int = 0
    """Number of warm up steps for the scheduler"""
    value_init_steps: int = 0
    """Number of steps to initialize the value"""
    value_lr_scheduler_type: str = "constant"
    """Which scheduler to use"""
    value_warm_up_steps: int = 0
    """Number of warm up steps for the scheduler"""
    policy_warmup_ratio: float = 0.0
    """Ratio of warmup steps to total steps (takes precedence over `policy_warm_up_steps`)"""
    value_warmup_ratio: float = 0.0
    """Ratio of warmup steps to total steps (takes precedence over `value_warm_up_steps`)"""
    max_grad_norm: float = 1.0
    """The maximum gradient norm"""
    max_approx_kl: float = 1e8
    """The maximum KL divergence"""

    # various batch sizes
    gradient_accumulation_steps: Optional[int] = None
    """The number of gradient accumulation steps"""
    per_device_train_batch_size: Optional[int] = 2
    """The forward batch size per device (local_micro_batch_size)"""
    # per_device_eval_batch_size: Optional[int] = 1
    # """The forward batch size per device for evaluation (local_micro_batch_size)"""
    total_episodes: Optional[int] = 100000
    """The total number of training episodes"""
    world_size: Optional[int] = None
    """The number of processes (GPUs) to use"""
    micro_batch_size: Optional[int] = None
    """The micro batch size across devices (HF's `per_device_train_batch_size` * `world_size`)"""
    local_rollout_batch_size: int = 2
    """The number of rollout episodes per iteration per device"""
    rollout_batch_size: Optional[int] = None
    """The number of rollout episodes per iteration"""
    num_training_steps: Optional[int] = None
    """The number of training_steps to train"""
    # num_evals: int = 4
    # """The number of evaluations to run throughout training"""
    # eval_freq: Optional[int] = None
    # """The frequency of evaluation steps"""
    # local_dataloader_batch_size: Optional[int] = None
    # """The batch size per GPU for the dataloader"""
    save_freq: int = -1
    """How many train steps to save the model"""
    num_epochs: int = 1
    """the number of epochs to train (set to 1 to prevent deviating from sft checkpoint)"""
    num_mini_batches: int = 1
    """Number of minibatches to split a batch into"""
    local_mini_batch_size: int = 64
    """the mini batch size per GPU"""
    mini_batch_size: Optional[int] = None
    """the mini batch size across GPUs"""
    local_rollout_forward_batch_size: int = 64
    """per rank no grad forward pass in the rollout phase"""

    # generation config
    response_length: int = 8
    """the length of the response"""
    stop_token_id: Optional[int] = None
    """the truncation token id"""
    min_response_length: int = 0
    """stop only after this many tokens"""
    temperature: float = 1.0
    """the sampling temperature"""
    verify_reward_value: float = 1.0
    """the reward value for responses that do not contain `stop_token_id`"""
    penalty_reward_value: float = -1.0
    """the reward value for responses that do not contain `stop_token_id`"""
    non_stop_penalty: bool = False
    """whether to penalize responses that do not contain `stop_token_id`"""
    number_envs_per_task: int = 1
    """the number of samples to generate per prompt, useful for easy-star"""

    # PPO specific args
    beta: float = 0.0
    """the beta value of the RLHF objective (KL coefficient)"""
    whiten_rewards: bool = False
    """whether to whiten the rewards"""
    cliprange: float = 0.2
    """the clip range"""
    vf_coef: float = 1.0
    """the value function coefficient"""
    cliprange_value: float = 0.2
    """the clip range for the value function"""
    gamma: float = 0.99
    """the discount factor (1.0 for sparse rewards, 0.99 for normal case)"""
    lam: float = 0.95
    """the lambda value for GAE (1.0 for sparse rewards, 0.95 for normal case)"""
    kl_estimator: str = "kl1"
    """the KL estimator to use"""
    process_reward_model: bool = False
    """the process reward model (prm), for dense reward"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    norm_adv: bool = False
    """Toggles advantages normalization"""

    # async setting
    async_mode: bool = False
    """Whether to run the generation in async mode which learns from the second latest policy like Cleanba (https://arxiv.org/abs/2310.00036)"""

    # Ray specific
    actor_num_gpus_per_node: List[int] = field(default_factory=lambda: [1])
    """number of gpus per node for actor learner"""

    # vLLM specific
    vllm_num_engines: int = 1
    """number of vLLM Engines, set to 0 to disable vLLM"""
    vllm_tensor_parallel_size: int = 1
    """tensor parallel size of vLLM Engine for multi-GPU inference (1 for single GPU inference)"""
    vllm_enforce_eager: bool = True
    """whether to enforce eager execution for vLLM, set to True to avoid building cuda graph"""
    vllm_sync_backend: str = "nccl"
    """FSDP -> vLLM weight sync backend"""
    enable_prefix_caching: bool = False
    """whether to enable prefix caching"""
    gpu_memory_utilization: float = 0.9
    """pre-allocated GPU memory utilization for vLLM"""
    gather_whole_model: bool = True
    """whether to gather the whole model to broadcast (not doable for 70B but can be faster for 8B)"""

    # FSDP-Specific Parameters
    sharding_strategy: str = "full-shard"
    """The sharding strategy to use"""
    offload: bool = False
    """Whether to offload the model to CPU to save GPU memory"""

    # wandb and HF tracking configs
    use_wandb: bool = False
    """If toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project: str = "openvla"
    """The wandb's project name"""
    wandb_entity: Optional[str] = "openvla_cvpr"
    """The entity (team) of wandb's project"""
    wandb_offline: bool = False
    """Whether to run wandb in offline mode"""
    run_id_note: Optional[str] = None
    """Extra note for logging, Weights & Biases"""
    push_to_hub: bool = True
    """Whether to upload the saved model to huggingface"""
    hf_entity: Optional[str] = None
    """The user or org name of the model repository from the Hugging Face Hub"""
    hf_repo_id: Optional[str] = None
    """The id of the saved model in the Hugging Face Hub (can be autoset if not given)"""
    hf_repo_revision: Optional[str] = None
    """The revision of the saved model in the Hugging Face Hub (can be autoset if not given)"""
    hf_repo_url: Optional[str] = None
    """The url of the saved model in the Hugging Face Hub (will be autoset)"""


def get_num_patches(image_size: int, patch_size: int) -> int:
    grid_length = image_size // patch_size
    return grid_length * grid_length


def calculate_runtime_args(args: Args,):
    """calculate (in-place) runtime args such as the effective batch size, word size, etc."""
    args.gradient_accumulation_steps = exact_div(
        args.local_mini_batch_size,
        args.per_device_train_batch_size,
        "`local_mini_batch_size` must be a multiple of `per_device_train_batch_size`",
    )
    args.world_size = sum(args.actor_num_gpus_per_node)
    args.micro_batch_size = int(args.per_device_train_batch_size * args.world_size)
    args.rollout_batch_size = int(args.local_rollout_batch_size * args.world_size)
    args.n_rollout_threads = args.local_rollout_batch_size
    args.num_tasks_per_suite = args.local_rollout_batch_size

    # assert args.num_tasks_per_suite == 10

    args.mini_batch_size = int(args.local_mini_batch_size * args.world_size)
    args.num_training_steps = args.total_episodes // (args.rollout_batch_size)
    args.train_batch_size = int(args.local_rollout_batch_size * args.num_steps)
    args.num_mini_batches = exact_div(args.rollout_batch_size * args.num_steps, args.mini_batch_size)

    # PPO logic: do checks and set up dataloader batch size
    if args.whiten_rewards:
        assert (
            args.local_mini_batch_size >= 8
        ), f"Per-rank minibatch size {args.local_mini_batch_size} is insufficient for whitening"

    exp_id = (
        f"ppo+{args.dataset_name}"
        f"+tasks{np.unique(args.task_ids).size}"
        f"+trials{args.num_trials_per_task}"
        f"+ns{args.num_steps}"
        f"+maxs{args.max_env_length}"
        f"+rb{args.rollout_batch_size}" # rollout batch size
        f"+tb{args.mini_batch_size * args.gradient_accumulation_steps}" # training batch size
        f"+lr-{args.learning_rate}"
        f"+vlr-{args.value_learning_rate}"
        f"+s-{args.seed}"
    )
    if args.run_id_note is not None:
        exp_id += f"--{args.run_id_note}"
    if args.use_lora:
        exp_id += f"+lora"
    if args.norm_adv:
        exp_id += f"+norm_adv"
    # if args.image_aug:
    #     exp_id += "--image_aug"
    args.exp_id = exp_id
    cprint(f"Experiment ID: {exp_id}", "green")

    args.unnorm_key = args.task_suite_name

def add_padding(sequences: List[List[int]], pad_token_id: int, length: int) -> List[List[int]]:
    for i in range(len(sequences)):
        sequences[i] = sequences[i] + [pad_token_id] * (length - len(sequences[i]))
    return sequences

class RayProcess:
    def __init__(self, world_size, rank, master_addr, master_port):
        logging.basicConfig(
            format="%(asctime)s %(levelname)-8s %(message)s",
            level=logging.INFO,
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self._world_size = world_size
        self._rank = rank
        self._master_addr = master_addr if master_addr else self._get_current_node_ip()
        self._master_port = master_port if master_port else self._get_free_port()
        os.environ["MASTER_ADDR"] = self._master_addr
        os.environ["MASTER_PORT"] = str(self._master_port)
        os.environ["WORLD_SIZE"] = str(self._world_size)
        os.environ["RANK"] = str(self._rank)
        # NOTE: Ray will automatically set the *_VISIBLE_DEVICES
        # environment variable for each actor, unless
        # RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES is set, so
        # set local rank to 0 when the flag is not applicable.
        os.environ["LOCAL_RANK"] = str(ray.get_gpu_ids()[0]) if ray_noset_visible_devices() else "0"

        random.seed(self._rank)
        np.random.seed(self._rank)
        torch.manual_seed(self._rank)

    @staticmethod
    def _get_current_node_ip():
        address = ray._private.services.get_node_ip_address()
        # strip ipv6 address
        return address.strip("[]")

    @staticmethod
    def _get_free_port():
        with socket.socket() as sock:
            sock.bind(("", 0))
            return sock.getsockname()[1]

    def get_master_addr_port(self):
        return self._master_addr, self._master_port

def get_sharding_strategy(device_mesh):
    if device_mesh.ndim == 1:
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif device_mesh.ndim == 2:
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    else:
        raise NotImplementedError(f"Get device mesh ndim={device_mesh.ndim}, but only support 1 or 2")
    return sharding_strategy

@ray.remote(num_gpus=1)
class PolicyTrainerRayProcess(RayProcess):
    def from_pretrained(self, args):
        """Initialize models and optimizers from pretrained checkpoints"""
        # Update logger with rank information
        global logger
        logger = init_logger(__name__, self._rank)
        # Register OpenVLA model to HF Auto Classes
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
        
        if getattr(args, "vllm_num_engines", 0) > 0:
            # To prevent hanging during NCCL synchronization of weights between deepspeed and vLLM.
            # see https://github.com/vllm-project/vllm/blob/c6b0a7d3ba03ca414be1174e9bd86a97191b7090/vllm/worker/worker_base.py#L445
            if getattr(args, "vllm_sync_backend", "nccl") == "nccl":
                os.environ["NCCL_CUMEM_ENABLE"] = "0"

        import torch.distributed
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")

        self.args = args

        # build device mesh for FSDP
        world_size = torch.distributed.get_world_size()
        print(f"[Actor] World size: {world_size}")
        self.device_mesh = init_device_mesh('cuda', mesh_shape=(world_size,), mesh_dim_names=['fsdp'])

        self._local_rank = int(os.environ["LOCAL_RANK"])
        # logger.info(f"[Actor] Local rank: {self._local_rank}")
        torch.cuda.set_device(self._local_rank)

        # Initialize base model
        torch_dtype = torch.bfloat16
        model = AutoModelForVision2Seq.from_pretrained(
            args.vla_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        # NOTE: 256 is the max image tokens for openvla
        self.hf_config = deepcopy(model.config)
        self.max_image_tokens = self.get_max_image_tokens()

        # Load adapter checkpoint if specified
        if args.load_adapter_checkpoint is not None:
            # Load dataset statistics if available
            dataset_statistics_path = os.path.join(args.load_adapter_checkpoint, "dataset_statistics.json")
            if os.path.isfile(dataset_statistics_path):
                with open(dataset_statistics_path, "r") as f:
                    norm_stats = json.load(f)
                model.norm_stats = norm_stats
            # Load adapter weights
            model = PeftModel.from_pretrained(
                model,
                args.load_adapter_checkpoint,
                is_trainable=True
            )
            logger.info("[Actor] Loaded from adapter checkpoint")
            model.print_trainable_parameters()
        # Initialize new LoRA if no checkpoint
        else:
            if args.use_lora:
                lora_config = LoraConfig(
                    r=args.lora_rank,
                    lora_alpha=min(args.lora_rank, 16),
                    lora_dropout=args.lora_dropout,
                    target_modules="all-linear",
                    init_lora_weights="gaussian",
                )
                if args.use_quantization:
                    model = prepare_model_for_kbit_training(model)
                model = get_peft_model(model, lora_config)
                
                model.print_trainable_parameters()
                logger.info("[Actor] Training from scratch with LoRA")
            else:
                logger.info("[Actor] Training from scratch")

        model.to(torch_dtype)
        disable_dropout_in_model(model)
        if args.enable_gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

        fsdp_sharding_strategy = get_sharding_strategy(self.device_mesh)
        log_gpu_memory_usage("[Actor] Before FSDP wrapping", rank=self._rank, logger=logger, level=logging.INFO)
        auto_wrap_policy = get_fsdp_wrap_policy_openvla(model, is_lora=args.use_lora)
        # auto_wrap_policy = get_fsdp_wrap_policy(model, is_lora=args.use_lora) # ~40GB memory
        logger.info(f'[Actor] wrap_policy: {auto_wrap_policy}')
        # cpu_offload = CPUOffload(offload_params=True) if args.offload else None
        cpu_offload = None  # NOTE:  We force turn off CPUOffload for critic because it causes incorrect results when using grad accumulation

        fsdp_precision_policy = MixedPrecision(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32
        )
        self.model = FSDP(
            model,
            cpu_offload=cpu_offload,
            param_init_fn=init_fn,
            use_orig_params=False,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
            sharding_strategy=fsdp_sharding_strategy,
            mixed_precision=fsdp_precision_policy,
            sync_module_states=True,
            device_mesh=self.device_mesh,
            forward_prefetch=False,
        )
        del model
        # self.model = model.to(device=torch.device("cuda"))    # w/o FSDP
        logger.info("[Actor] Initialized FSDP model")
        log_gpu_memory_usage("[Actor] After model init", rank=self._rank, logger=logger, level=logging.INFO)
        self.policy_optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            eps=args.eps,
            weight_decay=args.weight_decay,
        )
        num_training_steps = args.num_training_steps * args.num_epochs
        policy_warm_up_steps = args.policy_warm_up_steps
        if args.policy_warmup_ratio >= 0.0:
            policy_warm_up_steps = int(num_training_steps * args.policy_warmup_ratio)
        self.policy_scheduler = get_scheduler(
            name=args.policy_lr_scheduler_type,
            optimizer=self.policy_optimizer,
            num_warmup_steps=policy_warm_up_steps,
            num_training_steps=num_training_steps,
        )
        # Verify model contains required action normalization stats
        if args.unnorm_key not in self.model.norm_stats:
            if f"{args.unnorm_key}_no_noops" in self.model.norm_stats:
                args.unnorm_key = f"{args.unnorm_key}_no_noops"
            else:
                raise ValueError(f"Action un-norm key: {args.unnorm_key} not found in VLA `norm_stats`: {self.model.norm_stats.keys()}")
        log_gpu_memory_usage("[Actor] After fsdp wrapping", rank=self._rank, logger=logger, level=logging.INFO)

        # Initialize value model (critic)
        use_value_model = args.use_value_model
        if use_value_model:
            if args.value_model_type == "vla":
                value_model = AutoModelForVision2Seq.from_pretrained(
                        args.vla_path,
                        attn_implementation="flash_attention_2",
                        torch_dtype=torch_dtype,
                        low_cpu_mem_usage=True,
                        trust_remote_code=True,
                    )
                log_gpu_memory_usage("[Critic] After value model init", rank=self._rank, logger=logger, level=logging.INFO)
                if args.value_adapter_dir is not None:
                    value_model = PeftModel.from_pretrained(
                        value_model, 
                        args.value_adapter_dir,
                        is_trainable=True
                    )
                    logger.info("[Critic] Loaded from adapter checkpoint")
                elif args.value_use_lora:
                    lora_config = LoraConfig(
                        r=args.value_lora_rank,
                        lora_alpha=min(args.value_lora_rank, 16),
                        lora_dropout=args.value_lora_dropout,
                        target_modules="all-linear",
                        init_lora_weights="gaussian",
                    )
                    value_model = get_peft_model(value_model, lora_config)
                    logger.info("[Critic] Training from scratch with LoRA")
                else:
                    logger.info("[Critic] Training from scratch")
                value_model = CriticVLA(args, value_model, adapter_dir=args.value_model_path)
                auto_wrap_policy = get_fsdp_wrap_policy_openvla(value_model, is_lora=True)
            elif args.value_model_type == "film":
                text_encoder = AutoModel.from_pretrained(
                    "distilbert/distilbert-base-uncased", 
                    torch_dtype=torch.bfloat16, 
                    attn_implementation="flash_attention_2",
                )
                value_model = CriticFilm(text_encoder=text_encoder)
                logger.info("[Critic] Initialized FiLM model from scratch")
                # auto_wrap_policy = get_fsdp_wrap_policy(module=value_model)
                # auto_wrap_policy = get_fsdp_wrap_policy_film(module=value_model)
            else:
                raise ValueError(f"Value model: {args.value_model_type} not found")

            value_model.print_trainable_parameters()
            value_model.to(torch_dtype)
            disable_dropout_in_model(value_model)
            if args.enable_gradient_checkpointing and args.value_model_type == "vla":
                value_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

            logger.info(f'[Critic] wrap_policy: {auto_wrap_policy}')
            # self.value_model = value_model.to(device=torch.cuda.current_device())
            # fsdp_precision_policy = MixedPrecision(
            #     param_dtype=torch.float32, reduce_dtype=torch.float32, buffer_dtype=torch.float32
            # )
            self.value_model = FSDP(
                value_model,
                cpu_offload=cpu_offload,
                param_init_fn=init_fn,
                use_orig_params=False,
                device_id=torch.cuda.current_device(),
                auto_wrap_policy=auto_wrap_policy,
                sharding_strategy=fsdp_sharding_strategy,
                mixed_precision=fsdp_precision_policy,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                forward_prefetch=False,
            )
            del value_model
            log_gpu_memory_usage("[Critic] After value model FSDP wrapping", rank=self._rank, logger=logger, level=logging.INFO)
            self.value_optimizer = torch.optim.AdamW(
                self.value_model.parameters(),
                lr=args.value_learning_rate,
                eps=args.eps,
                weight_decay=args.weight_decay,
            )
            value_warm_up_steps = args.value_warm_up_steps
            if args.value_warmup_ratio >= 0.0:
                value_warm_up_steps = int(num_training_steps * args.value_warmup_ratio)
            self.value_scheduler = get_scheduler(
                name=args.value_lr_scheduler_type,
                optimizer=self.value_optimizer,
                num_warmup_steps=value_warm_up_steps,
                num_training_steps=num_training_steps,
            )
        torch.cuda.empty_cache()
        log_gpu_memory_usage("After all init", rank=self._rank, logger=logger, level=logging.INFO)

    def get_max_image_tokens(self) -> int:
        hf_config = self.hf_config
        backbone_id = hf_config.vision_backbone_id
        if backbone_id.startswith("dinosiglip"):
            timm_model_ids = VISION_BACKBONE_TO_TIMM_ID[backbone_id]    # e.g., ["vit_large_patch14_reg4_dinov2.lvd142m", "vit_so400m_patch14_siglip_224"]
            image_size = hf_config.image_sizes[0]
            patch_size = int(timm_model_ids[0].split("patch")[1].split("_")[0])   # HACK: get patch_size from timm_model_ids
            num_image_tokens = get_num_patches(image_size, patch_size)
        else:
            raise NotImplementedError(f"Unsupported vision backbone: {backbone_id}; only dinosiglip is supported.")
        return num_image_tokens

    def train(
        self,
        processor: ProcessorMixin,
        vllm_engines: List[ray.actor.ActorHandle],
        metrics_queue: Queue,
    ):
        """Main training loop for PPO"""
        logger.info("Starting training loop")
        torch.set_printoptions(precision=6, sci_mode=False)

        args = self.args
        hf_config = deepcopy(self.model.config)
        timer = TimingManager()

        # Environment
        local_rollout_indices = slice(self._rank * args.local_rollout_batch_size, (self._rank + 1) * args.local_rollout_batch_size)

        args.task_ids = args.task_ids[local_rollout_indices]
        args.env_gpu_id = self._rank
        logger.info(f"Current Device ID: {self._rank}; Task IDs: {args.task_ids}")
        train_envs = VLAEnv(cfg=args, mode="train")
        action_dim = train_envs.action_space[0]

        padding_side = "right"

        accelerator = Namespace()
        accelerator.process_index = self._rank
        accelerator.num_processes = self._world_size
        accelerator.is_main_process = self._rank == 0
        dist.barrier()
        if accelerator.is_main_process:
            master_address = ray._private.services.get_node_ip_address()
            logger.info(f"Master address: {master_address}")
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]
            logger.info(f"Master port: {master_port}")
            vllm_num_engines, vllm_tensor_parallel_size = (
                args.vllm_num_engines,
                args.vllm_tensor_parallel_size,
            )
            world_size = vllm_num_engines * vllm_tensor_parallel_size + 1
            backend = getattr(args.vllm_sync_backend, "vllm_sync_backend", "nccl")
            group_name = "vllm-inference-group"
            refs = [
                engine.init_process_group.remote(
                    master_address=master_address,
                    master_port=master_port,
                    rank_offset=i * vllm_tensor_parallel_size + 1,
                    world_size=world_size,
                    group_name=group_name,
                    backend=backend,
                    use_ray=False,
                )
                for i, engine in enumerate(vllm_engines)
            ]
            logger.info(f"[vLLM] Initialized vLLM engines with group name: {group_name}")
            self.model_update_group = init_process_group(
                backend=backend,
                init_method=f"tcp://{master_address}:{master_port}",
                world_size=world_size,
                rank=0,
                group_name=group_name,
            )
            ray.get(refs)
            logger.info("[vLLM] Initialized vLLM engines")
        dist.barrier()

        def _broadcast_to_vllm():
            use_prefix_cache = getattr(args, "enable_prefix_caching", False)
            cache_reset_refs = []
            if use_prefix_cache and torch.distributed.get_rank() == 0:
                # clear prefix cache
                for engine in vllm_engines:
                    cache_reset_refs.append(engine.reset_prefix_cache.remote())

            torch.cuda.empty_cache()
            # model = self.model.module
            model = self.model

            # Get parameter names first without loading full params
            param_names = []
            if is_peft_model(model):
                # Get list of parameter names that will need to be broadcast
                with torch.no_grad():
                    for name, _ in model.named_parameters():
                        processed_name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                        if model.prefix not in processed_name and "original_module" not in processed_name:
                            if processed_name.startswith("modules_to_save.default."):
                                processed_name = processed_name.replace("modules_to_save.default.", "")
                            param_names.append(processed_name)
            else:
                for name, _ in model.named_parameters():
                    param_names.append(name)
                    
            num_params = len(param_names)
            # Process parameters in batches to reduce memory usage
            # batch_size = 8    # low memory
            batch_size = num_params
            print(f"🔥🔥🔥 Broadcasting {num_params} parameters in batches of {batch_size}")
            
            # Process in batches
            for batch_start in range(0, num_params, batch_size):
                batch_end = min(batch_start + batch_size, num_params)
                batch_names = param_names[batch_start:batch_end]
                
                # For this batch only, summon full parameters
                with FSDP.summon_full_params(model, writeback=False):   # this takes up a lot of memory
                    if is_peft_model(model):
                        # Create a modified copy of the model for broadcasting
                        # instead of directly modifying the original model
                        with torch.no_grad():
                            if batch_start == 0:  # Only merge once at the beginning
                                logger.info("Merging adapter")
                                model.merge_adapter()
                            
                            # Get only the batch parameters instead of full state dict
                            batch_state_dict = {}
                            for name, param in model.named_parameters():
                                processed_name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                                if processed_name in batch_names or name in batch_names:
                                    if model.prefix not in processed_name and "original_module" not in processed_name:
                                        if processed_name.startswith("modules_to_save.default."):
                                            key = processed_name.replace("modules_to_save.default.", "")
                                            batch_state_dict[key] = param.data
                                        else:
                                            batch_state_dict[processed_name] = param.data
                            
                            if batch_end == num_params:  # Only unmerge at the end
                                logger.info("Unmerging adapter")
                                model.unmerge_adapter()
                    else:
                        # Get only the batch parameters
                        batch_state_dict = {}
                        for name, param in model.named_parameters():
                            if name in batch_names:
                                batch_state_dict[name] = param.data

                    # Broadcast this batch of parameters
                    for name, param in batch_state_dict.items():
                        is_last_param = (batch_end == num_params and name == list(batch_state_dict.keys())[-1])
                        
                        # Fire all vllm engines for broadcast
                        if torch.distributed.get_rank() == 0:
                            shape = param.shape
                            refs = [
                                engine.update_weight.remote(
                                    name, dtype=param.dtype, shape=shape, empty_cache=is_last_param
                                )
                                for engine in vllm_engines
                            ]
                        
                            # Only broadcast from rank 0
                            torch.distributed.broadcast(param.data, 0, group=self.model_update_group)
                            ray.get(refs)
                            refs = None  # Clear refs to free memory
                
                # Explicit cleanup after each batch
                batch_state_dict = None
                torch.cuda.empty_cache()
                print(f"Processed batch {batch_start//batch_size + 1}/{(num_params-1)//batch_size + 1}")
            
            if cache_reset_refs:
                ray.get(cache_reset_refs)
                
            torch.cuda.empty_cache()
            torch.distributed.barrier()

        if not args.debug:
            with timer.timer("broadcast"):
                _broadcast_to_vllm()

        args.stop_token_id = processor.tokenizer.eos_token_id
        args.bos_token_id = processor.tokenizer.bos_token_id
        args.pad_token_id = processor.tokenizer.pad_token_id

        generation_config = SamplingParams(
            temperature=args.temperature,
            top_p=1.0,
            max_tokens=args.response_length,
            include_stop_str_in_output=False,
            detokenize=False,
            n=1,
            # seed=args.seed,   # NOTE: this will lead to deterministic results
            logprobs=1,
        )
        # logger.info("setup async queues")
        response_ids_Q = Queue(maxsize=1)
        param_prompt_Q = Queue(maxsize=1)

        def vllm_generate(
            generation_config: SamplingParams,
            response_ids_Q: Queue,
            param_prompt_Q: Queue,
            num_training_steps: int,
            resume_training_step: int,
        ):
            llm = vllm_engines[0]
            while True:
                g_queries_list = param_prompt_Q.get()
                if g_queries_list is None:
                    break
                prompt_token_ids = g_queries_list["input_ids"]
                pixel_values = g_queries_list["pixel_values"]
                pixel_values_uint8 = (pixel_values).astype(np.uint8)  # [B, H, W, C], already in [0, 255]
                
                llm_inputs = []
                for i in range(len(prompt_token_ids)):
                    fixed_prompt_token_ids = [item for item in prompt_token_ids[i].tolist() if item != args.pad_token_id]
                    fixed_prompt_token_ids = fixed_prompt_token_ids[0:1] + [args.pad_token_id] * 256 + fixed_prompt_token_ids[1:]
                    # logger.info(f"🔥🔥🔥 Prompt: {fixed_prompt_token_ids}")
                    # prompt = processor.decode(prompt_token_ids[i].tolist(), skip_special_tokens=True).strip()
                    # prompt = "<PAD>" + prompt + "▁"
                    # logger.info(f"🔥🔥🔥 Prompt: {prompt}")
                    pil_img = Image.fromarray(pixel_values_uint8[i])
                    llm_inputs.append({
                        "prompt_token_ids": fixed_prompt_token_ids,
                        # "prompt": prompt,
                        "multi_modal_data": {"image": pil_img},
                    })
                generation_start_time = time.time()
                actions, response_ids, response_logprobs = ray.get(
                    llm.predict_action.remote(
                        llm_inputs,
                        sampling_params=generation_config, 
                        use_tqdm=False,
                        unnorm_key=args.unnorm_key,
                        )
                )
                print(f"🔥🔥🔥 Action generation time: {time.time() - generation_start_time:.2f} seconds")
                response_ids_Q.put((actions, response_ids, response_logprobs))

        resume_training_step = 1
        if accelerator.is_main_process:
            thread = threading.Thread(
                target=vllm_generate,
                args=(
                    generation_config,
                    response_ids_Q,
                    param_prompt_Q,
                    args.num_training_steps,
                    resume_training_step,
                ),
            )
            thread.start()
            logger.info("[vLLM] vllm generate thread starts")

        device = torch.device(self._local_rank)

        num_channels = 3
        if args.model_family == "openvla":
            num_channels = num_channels * 2 # stack for dinosiglip
        image_height, image_width = hf_config.image_sizes
        
        logger.info(f"🕹️🕹️🕹️ Env reset")
        local_obs, _ = train_envs.reset()
        # ------------------------------------------------------------
        # local_obs['prompts']: List[str]
        # local_obs['pixel_values']: List[Image.Image]
        # ------------------------------------------------------------

        # Gather observations from all GPUs
        world_size = dist.get_world_size()

        # LOGIC: get local tensor obs for gather, gather to all GPUs, then concat to global tensor obs to call vllm engines
        local_token_obs = {
            "input_ids": torch.ones(args.local_rollout_batch_size, args.context_length - 1, device=device, dtype=torch.float32) * args.pad_token_id,
            "pixel_values": torch.zeros(args.local_rollout_batch_size, num_channels, image_height, image_width, device=device, dtype=torch.float32),
        }
        processed_obs = process_with_padding_side(processor, local_obs["prompts"], local_obs["pixel_values"], padding=True, padding_side=padding_side).to(device, dtype=torch.float32)
        local_token_obs["input_ids"][:, :processed_obs["input_ids"].shape[1]] = processed_obs["input_ids"]
        local_token_obs["input_ids"] = add_special_token(local_token_obs["input_ids"], pad_token_id=args.pad_token_id)
        local_token_obs["pixel_values"][:] = processed_obs["pixel_values"]
        del processed_obs

        gathered_input_ids = [torch.zeros_like(local_token_obs["input_ids"]) for _ in range(world_size)]
        # logger.info(f"world_size: {world_size}")
        # NOTE: we use the un-processed pixel values to call vllm engines
        pixel_array = np.stack([np.array(img) for img in local_obs["pixel_values"]])    # [B, H, W, C]
        original_pixel_values_tensor = torch.from_numpy(pixel_array).to(device, dtype=torch.float32)    # .permute(0, 3, 1, 2)
        gathered_pixel_values = [torch.zeros_like(original_pixel_values_tensor) for _ in range(world_size)]
        dist.all_gather(gathered_input_ids, local_token_obs["input_ids"])
        dist.all_gather(gathered_pixel_values, original_pixel_values_tensor)
        dist.barrier()
        # Create global observation by concatenating all gathered data
        global_token_obs = {
            "input_ids": torch.cat(gathered_input_ids, dim=0).to(dtype=torch.long).cpu().numpy(),
            "pixel_values": torch.cat(gathered_pixel_values, dim=0).cpu().numpy(),
        }
        # allocate the obs to rank0 to call vllm engines
        if accelerator.is_main_process:
            param_prompt_Q.put(global_token_obs)

        
        local_rewards = None
        local_dones = torch.zeros(args.local_rollout_batch_size, device=device, dtype=torch.float32)
        local_actions = None
        local_token_obs["input_ids"] = local_token_obs["input_ids"].to(dtype=torch.long)
        queries_next = local_token_obs["input_ids"]
        pixel_values_next = local_token_obs["pixel_values"]
        dones_next = local_dones

        # set up rollout storage
        g_vllm_responses = torch.zeros(
            (args.rollout_batch_size, args.response_length),
            device=device,
            dtype=torch.long,
        )
        g_vllm_logprobs = torch.zeros(
            (args.rollout_batch_size, args.response_length),
            device=device,
            dtype=torch.float32,
        )
        global_actions = torch.zeros(
            (args.rollout_batch_size, action_dim),
            device=device,
            dtype=torch.float32,
        )
        # set up training stats
        stats_shape = (
            args.num_epochs,
            args.num_mini_batches,
            args.gradient_accumulation_steps,
        )
        approxkl_stats = torch.zeros(stats_shape, device=device)
        pg_clipfrac_stats = torch.zeros(stats_shape, device=device)
        pg_loss_stats = torch.zeros(stats_shape, device=device)
        pg_grad_norm_stats = torch.zeros(stats_shape, device=device)
        vf_clipfrac_stats = torch.zeros(stats_shape, device=device)
        vf_loss_stats = torch.zeros(stats_shape, device=device)
        vf_grad_norm_stats = torch.zeros(stats_shape, device=device)
        ratio_stats = torch.zeros(stats_shape, device=device)
        local_metrics = {}
        episode = args.rollout_batch_size * (resume_training_step - 1)   # NOTE: we start from 1

        # ALGO Logic: Storage setup
        queries = torch.ones(
            (args.num_steps, args.local_rollout_batch_size, args.context_length),
            device=device,
            dtype=torch.long,
        ) * args.pad_token_id
        pixel_values = torch.zeros(
            # (args.num_steps, args.rollout_batch_size) + train_envs.observation_space["pixel_values"],
            (args.num_steps, args.local_rollout_batch_size, num_channels, image_height, image_width),
            device=device,
            dtype=torch.float32,
        )
        responses = torch.zeros(
            (args.num_steps, args.local_rollout_batch_size, args.response_length),
            device=device,
            dtype=torch.long,
        )
        logprobs = torch.zeros(
            (args.num_steps, args.local_rollout_batch_size, args.response_length),
            device=device,
            dtype=torch.float32,
        )
        vllm_logprobs = torch.zeros(
            (args.num_steps, args.local_rollout_batch_size, args.response_length),
            device=device,
            dtype=torch.float32,
        )
        scores = torch.zeros(
            (args.num_steps, args.local_rollout_batch_size),
            device=device,
            dtype=torch.float32,
        )
        dones = torch.zeros(
            (args.num_steps, args.local_rollout_batch_size),
            device=device,
            dtype=torch.bool,
        )
        values = torch.zeros(
            (args.num_steps, args.local_rollout_batch_size),
            device=device,
            dtype=torch.float32,
        )
        actions = None
        g_response_token_ids = None
        g_response_logprobs = None
        log_gpu_memory_usage("[Rollout] Storage setup, before rollout", rank=accelerator.process_index, logger=logger, level=logging.INFO)

        # self.save_model(self.model, processor, args.exp_dir)

        resume_training_step = 1
        global_step = 0
        # episodic_returns = Queue(maxsize=args.rollout_batch_size)
        # episodic_lengths = Queue(maxsize=args.rollout_batch_size)
        # Begin training loop
        for training_step in range(resume_training_step, args.num_training_steps):
            episodic_returns = []
            episodic_lengths = []
            episode += args.rollout_batch_size  # rollout batch size is the number of parallel environments

            if training_step != 1:
                # sync the policy model after each num_steps
                start_time = time.time()
                if not args.debug:
                    with timer.timer("broadcast"):
                        _broadcast_to_vllm()
                if accelerator.is_main_process: #and args.verbose:
                    print(
                        f"🔥🔥🔥 Syncing weights using shared memory; Time to sync weights: {time.time() - start_time:.2f} seconds"
                    )

            for step in range(args.num_steps):
                global_step += args.rollout_batch_size
                queries[step, :, :queries_next.shape[1]] = queries_next
                pixel_values[step] = pixel_values_next
                dones[step] = dones_next
            
                # ================= ROLLOUT PHASE =================
                # All inference during rollout is performed in no_grad mode.
                # We get logprobs, scores, values, and advantages in the forward pass.
                with torch.no_grad():
                    # Retrieve responses generated by the vLLM thread via a queue
                    g_vllm_responses = torch.zeros(args.rollout_batch_size, args.response_length, device=device, dtype=torch.float32)
                    g_vllm_logprobs = torch.zeros(args.rollout_batch_size, args.response_length, device=device, dtype=torch.float32)
                    global_actions = torch.zeros(args.rollout_batch_size, action_dim, device=device, dtype=torch.float32)
                    if accelerator.is_main_process:
                        with timer.timer("vllm_generate"):
                            response_data: tuple[np.ndarray, list[list[int]], list[list[float]]] = response_ids_Q.get()
                            actions, g_response_token_ids, g_response_logprobs = response_data
                        
                        global_actions = torch.tensor(actions, device=device, dtype=torch.float32)
                        g_vllm_responses[:] = torch.tensor(add_padding(g_response_token_ids, args.pad_token_id, args.response_length), device=device, dtype=torch.float32)
                        g_vllm_logprobs[:] = torch.tensor(add_padding(g_response_logprobs, 0.0, args.response_length), device=device, dtype=torch.float32)
                    with timer.timer("broadcast"):
                        dist.broadcast(g_vllm_responses, src=0)
                        dist.broadcast(g_vllm_logprobs, src=0)
                        dist.broadcast(global_actions, src=0)
                    dist.barrier()
                    g_vllm_responses = g_vllm_responses.to(torch.long)
                    local_vllm_responses = g_vllm_responses[local_rollout_indices]
                    local_vllm_logprobs = g_vllm_logprobs[local_rollout_indices]
                    local_actions = global_actions[local_rollout_indices].cpu().numpy()

                    responses[step] = local_vllm_responses
                    # logprobs[step] = local_vllm_logprobs
                    vllm_logprobs[step] = local_vllm_logprobs

                    # Process mini-batches over the rollout data
                    self.model.eval() # verl
                    if args.use_value_model:
                        self.value_model.eval()
                    for i in range(0, args.local_rollout_batch_size, args.local_rollout_forward_batch_size):
                        query = queries[step, i : i + args.local_rollout_forward_batch_size]
                        response = responses[step, i : i + args.local_rollout_forward_batch_size]
                        pixel_value = pixel_values[step, i : i + args.local_rollout_forward_batch_size]

                        if padding_side == "right":
                            last_context_length = first_true_indices(query == args.pad_token_id)
                            query_response = torch.ones(query.shape[0], query.shape[1] + response.shape[1], device=device, dtype=torch.long) * args.pad_token_id
                            context_length = last_context_length + self.max_image_tokens
                            for j in range(query.shape[0]):
                                query_response[j, :last_context_length[j]] = query[j, :last_context_length[j]]
                                query_response[j, last_context_length[j]:last_context_length[j] + response.shape[1]] = response[j]
                        else:
                            query_response = torch.cat((query, response), dim=1)
                            context_length = query.shape[1] + self.max_image_tokens

                        # Get value estimates from the value model
                        start_time = time.time()
                        if args.use_value_model:
                            with timer.timer("value"):
                                value = get_reward(self.value_model, query, pixel_value, args.pad_token_id)
                        else:
                            value = torch.zeros(args.local_rollout_forward_batch_size, device=device)
                        torch.cuda.empty_cache()
                        logger.info(f"Value time: {time.time() - start_time} seconds")

                        # TODO: Argh... this is redundant with vllm logprobs. Try to remove it.
                        # start_time = time.time()
                        # with timer.timer("forward"):
                        #     logprob, logits = forward(
                        #         self.model, query_response, pixel_value, response,
                        #         args.pad_token_id, context_length, args.temperature
                        #     )
                        # torch.cuda.empty_cache()
                        # logger.info(f"Forward time: {time.time() - start_time} seconds")
                        # breakpoint()

                        # Compute a score using the reward model
                        # score = torch.zeros(query.shape[0], device=device)
                        # if args.process_reward_model:
                        #     processed_score = get_reward(self.reward_model, query_response, args.pad_token_id, context_length)
                        #     score += processed_score
                        
                        # Accumulate rollout data
                        # logprobs[step, i : i + args.local_rollout_forward_batch_size] = logprob
                        # scores[step, i : i + args.local_rollout_forward_batch_size] = score
                        values[step, i : i + args.local_rollout_forward_batch_size] = value

                    del query, response, pixel_value, value
                    gc.collect()
                    torch.cuda.empty_cache()

                local_token_obs = {
                    "input_ids": torch.ones(args.local_rollout_batch_size, args.context_length - 1, device=device, dtype=torch.float32) * args.pad_token_id,
                    "pixel_values": torch.zeros(args.local_rollout_batch_size, num_channels, image_height, image_width, device=device, dtype=torch.float32),
                }
                logger.info(f"🕹️🕹️🕹️ Env {step=}")
                with timer.timer("env_step"):
                    local_obs, local_rewards, local_dones, local_infos = train_envs.step(
                        local_actions, 
                        values=values[step].detach().cpu().numpy(), 
                        log_probs=vllm_logprobs[step].detach().cpu().numpy()
                    )
                processed_obs = process_with_padding_side(processor, local_obs["prompts"], local_obs["pixel_values"], padding=True, padding_side=padding_side).to(device, dtype=torch.float32)
                local_token_obs["input_ids"][:, :processed_obs["input_ids"].shape[1]] = processed_obs["input_ids"]
                local_token_obs["input_ids"] = add_special_token(local_token_obs["input_ids"], pad_token_id=args.pad_token_id)
                local_token_obs["pixel_values"][:] = processed_obs["pixel_values"]
                del processed_obs

                gathered_input_ids = [torch.zeros_like(local_token_obs["input_ids"]) for _ in range(world_size)]
                # convert local_obs["pixel_values"] to tensor
                # NOTE: we use the un-processed pixel values to call vllm engines
                pixel_array = np.stack([np.array(img) for img in local_obs["pixel_values"]])    # [B, H, W, C]
                original_pixel_values_tensor = torch.from_numpy(pixel_array).to(device, dtype=torch.float32)    # .permute(0, 3, 1, 2)
                dist.all_gather(gathered_input_ids, local_token_obs["input_ids"])
                dist.all_gather(gathered_pixel_values, original_pixel_values_tensor)
                dist.barrier()
                global_token_obs = {
                    "input_ids": torch.cat(gathered_input_ids, dim=0).to(dtype=torch.long).cpu().numpy(),
                    "pixel_values": torch.cat(gathered_pixel_values, dim=0).cpu().numpy(),
                }
                if accelerator.is_main_process:
                    param_prompt_Q.put(global_token_obs)

                local_rewards = torch.tensor(local_rewards, device=device, dtype=torch.float32)
                local_dones = torch.tensor(local_dones, device=device, dtype=torch.float32)
                scores[step] = local_rewards

                # compute episodic reward
                for i in range(args.local_rollout_batch_size):
                    if local_dones[i]:
                        episodic_returns.append(local_rewards[i].item())
                        episodic_lengths.append(local_infos["step_count_tmp"][i])

                local_token_obs["input_ids"] = local_token_obs["input_ids"].to(dtype=torch.long)
                queries_next = local_token_obs["input_ids"]
                pixel_values_next = local_token_obs["pixel_values"]
                dones_next = local_dones

                del local_token_obs
                torch.cuda.empty_cache()

            # if args.debug:
            #     continue

            # logger.info('gae')
            # compute advantages and returns
            with torch.no_grad() and timer.timer("gae"):    # TODO: optimize this
                if args.use_value_model:
                    self.value_model.eval()
                    next_value = torch.zeros(args.local_rollout_batch_size, device=device)
                    for i in range(0, args.local_rollout_batch_size, args.local_rollout_forward_batch_size):
                        query = queries_next[i : i + args.local_rollout_forward_batch_size]
                        pixel_value = pixel_values_next[i : i + args.local_rollout_forward_batch_size]
                        with torch.no_grad():
                            next_value[i : i + args.local_rollout_forward_batch_size] = get_reward(self.value_model, query, pixel_value, args.pad_token_id)
                else:
                    next_value = torch.zeros(args.local_rollout_batch_size, device=device)

                advantages = torch.zeros_like(scores).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - dones_next.float()  # Convert boolean to float
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1].float()
                        nextvalues = values[t + 1]
                    delta = scores[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.lam * nextnonterminal * lastgaelam
                returns = advantages + values
            torch.cuda.empty_cache()
            
            # if accelerator.is_main_process:
            #     logger.info(f"{dones=}")

            # flatten the batch: [num_steps, num_envs, ...] -> [num_steps * num_envs, ...]
            b_queries = queries.reshape(-1, *queries.shape[2:])
            b_pixel_values = pixel_values.reshape(-1, *pixel_values.shape[2:])
            b_responses = responses.reshape(-1, *responses.shape[2:])
            b_logprobs = logprobs.reshape(-1, *logprobs.shape[2:])
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Training phase
            log_gpu_memory_usage("[Training] Before training", rank=accelerator.process_index, logger=logger, level=logging.INFO)
            self.model.train()
            if args.use_value_model:
                self.value_model.train()
            with timer.timer("train_loop"):
                for epoch_idx in range(args.num_epochs):
                    b_inds = np.random.permutation(args.train_batch_size)   # each thread has its own permutation, dealing with its own local rollout batch
                    minibatch_idx = 0
                    for mini_batch_start in range(
                        0, args.train_batch_size, args.local_mini_batch_size
                    ):
                        mini_batch_end = mini_batch_start + args.local_mini_batch_size
                        mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                        gradient_accumulation_idx = 0
                        # TODO: gradient accumulation
                        for micro_batch_start in range(0, args.local_mini_batch_size, args.per_device_train_batch_size):
                            # logger.info(f"micro batch start: {micro_batch_start}")
                            micro_batch_end = micro_batch_start + args.per_device_train_batch_size
                            micro_batch_inds = mini_batch_inds[micro_batch_start:micro_batch_end]
                            mb_advantage = b_advantages[micro_batch_inds]
                            if args.norm_adv and mb_advantage.shape[0] > 8:
                                mb_advantage = (mb_advantage - mb_advantage.mean()) / (mb_advantage.std() + 1e-8)
                            mb_responses = b_responses[micro_batch_inds]
                            mb_queries = b_queries[micro_batch_inds]
                            mb_pixel_values = b_pixel_values[micro_batch_inds]
                            # mb_logprobs = b_logprobs[micro_batch_inds]
                            mb_return = b_returns[micro_batch_inds]
                            mb_values = b_values[micro_batch_inds]

                            if args.use_value_model:
                                vpred = get_reward(
                                    self.value_model, mb_queries, mb_pixel_values, args.pad_token_id
                                )
                                vpredclipped = torch.clamp(
                                    vpred,
                                    mb_values - args.cliprange_value,
                                    mb_values + args.cliprange_value,
                                )
                                # check if other thing needs gradient
                                # logger.info(f"vpred: {vpred.requires_grad}, mb_return_detached: {mb_return_detached.requires_grad}")
                                vf_losses1 = torch.square(vpred - mb_return)
                                vf_losses2 = torch.square(vpredclipped - mb_return)
                                vf_loss_max = torch.max(vf_losses1, vf_losses2)
                                vf_loss = 0.5 * vf_loss_max.mean() * args.vf_coef
                                
                                self.value_optimizer.zero_grad()
                                vf_loss.backward()
                                if isinstance(self.value_model, FSDP):
                                    value_grad_norm = self.value_model.clip_grad_norm_(max_norm=args.max_grad_norm)
                                else:
                                    value_grad_norm = torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), max_norm=args.max_grad_norm)
                                self.value_optimizer.step()
                                self.value_scheduler.step()

                            if padding_side == "right":
                                last_context_length = first_true_indices(mb_queries == args.pad_token_id)
                                mb_query_responses = torch.ones(mb_queries.shape[0], mb_queries.shape[1] + mb_responses.shape[1], device=device, dtype=torch.long) * args.pad_token_id
                                context_length = last_context_length + self.max_image_tokens
                                for j in range(mb_queries.shape[0]):
                                    mb_query_responses[j, :last_context_length[j]] = mb_queries[j, :last_context_length[j]]
                                    mb_query_responses[j, last_context_length[j]:last_context_length[j] + mb_responses.shape[1]] = mb_responses[j]
                            else:
                                mb_query_responses = torch.cat((mb_queries, mb_responses), dim=1)
                                context_length = mb_queries.shape[1] + self.max_image_tokens
    
                            if training_step > args.value_init_steps:
                                new_logprobs, new_logits = forward(
                                    self.model, mb_query_responses, mb_pixel_values, mb_responses, 
                                    args.pad_token_id, context_length, args.temperature
                                )
                                if epoch_idx == 0:
                                    # This can avoid additional forward pass in the rollout phase to get old logprobs. 
                                    # See the following blog post for more details:
                                    # https://costa.sh/blog-understanding-why-there-isn't-a-log-probability-in-trpo-and-ppo's-objective
                                    b_logprobs[micro_batch_inds] = new_logprobs.detach()
                                # NOTE: action logprobs = sum(logprobs)
                                mb_logprobs = b_logprobs[micro_batch_inds]  # old logprobs
                                new_logprobs = torch.sum(new_logprobs, dim=-1)
                                mb_logprobs = torch.sum(mb_logprobs, dim=-1)
                                logprobs_diff = new_logprobs - mb_logprobs
                                ratio = torch.exp(logprobs_diff)
                                pg_losses = -mb_advantage * ratio
                                pg_losses2 = -mb_advantage * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
                                pg_loss = torch.max(pg_losses, pg_losses2).mean()
                                
                                self.policy_optimizer.zero_grad()
                                pg_loss.backward()

                                with torch.no_grad():
                                    approxkl = ((-logprobs_diff).exp() - 1 + logprobs_diff).mean()  # kl3
                                    if approxkl > args.max_approx_kl or torch.isnan(approxkl):
                                        logger.info("--------------------------------")
                                        logger.info(f"Stopping training due to high KL divergence: {approxkl=}, {approxkl.dtype=}")
                                        logger.info(f"{mb_query_responses=}")
                                        # [B, C, H, W] -> [B]
                                        logger.info(f"{mb_pixel_values.mean(dim=(1, 2, 3))=}")
                                        logger.info(f"{new_logprobs=}, {new_logprobs.dtype=}")
                                        logger.info(f"{mb_logprobs=}, {mb_logprobs.dtype=}")
                                        logger.info(f"{mb_responses=}, {mb_responses.dtype=}")
                                        logger.info(f"{new_logits.argmax(dim=-1)=}, {new_logits.argmax(dim=-1).dtype=}")
                                        logger.info("--------------------------------")
                                        self.policy_optimizer.zero_grad()
                                    else:
                                        logger.info(f"Passed! {approxkl=}")

                                if isinstance(self.model, FSDP):
                                    policy_grad_norm = self.model.clip_grad_norm_(max_norm=args.max_grad_norm)
                                else:
                                    policy_grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=args.max_grad_norm)
                                # logger.info(f"Policy gradient norm: {policy_grad_norm}")
                                self.policy_optimizer.step()
                                self.policy_scheduler.step()
                            with torch.no_grad():
                                if args.use_value_model:
                                    vf_clipfrac = (vf_losses2 > vf_losses1).float().mean()
                                    vf_loss_stats[epoch_idx, minibatch_idx, gradient_accumulation_idx] = vf_loss
                                    vf_clipfrac_stats[epoch_idx, minibatch_idx, gradient_accumulation_idx] = vf_clipfrac
                                    vf_grad_norm_stats[epoch_idx, minibatch_idx, gradient_accumulation_idx] = value_grad_norm

                                if training_step > args.value_init_steps:
                                    pg_clipfrac = (pg_losses2 > pg_losses).float().mean()
                                    approxkl_stats[epoch_idx, minibatch_idx, gradient_accumulation_idx] = approxkl
                                    pg_clipfrac_stats[epoch_idx, minibatch_idx, gradient_accumulation_idx] = pg_clipfrac
                                    pg_loss_stats[epoch_idx, minibatch_idx, gradient_accumulation_idx] = pg_loss
                                    pg_grad_norm_stats[epoch_idx, minibatch_idx, gradient_accumulation_idx] = policy_grad_norm
                                    ratio_stats[epoch_idx, minibatch_idx, gradient_accumulation_idx] = ratio.mean()

                            gradient_accumulation_idx += 1
                        minibatch_idx += 1
                        # fmt: off
                        del mb_advantage, mb_responses, mb_query_responses, mb_return, mb_values
                        if training_step > args.value_init_steps:
                            del new_logprobs, logprobs_diff, ratio, pg_losses, pg_losses2, pg_loss
                        if args.use_value_model:
                            del vpred, vpredclipped, vf_losses1, vf_losses2, vf_loss_max
                        # fmt: on
                        # del everything and empty cache
                        torch.cuda.empty_cache()
                    del b_inds, mini_batch_inds

            # Update metrics
            # logger.info("start metrics")
            with torch.no_grad():
                local_metrics["objective/entropy"] = (-b_logprobs).sum(1).mean()
                local_metrics["objective/entropy_vllm"] = (-vllm_logprobs).sum(1).mean()
                local_metrics["objective/scores"] = scores.mean()
                local_metrics["objective/scores_std"] = scores.std() if scores.shape[0] > 1 else torch.tensor(0, device=device)
                local_metrics["objective/advantage_avg"] = advantages.mean()
                local_metrics["objective/advantage_std"] = advantages.std() if advantages.shape[0] > 1 else torch.tensor(0, device=device)
                local_metrics["policy/approxkl_avg"] = approxkl_stats.mean()
                local_metrics["policy/clipfrac_avg"] = pg_clipfrac_stats.mean()
                local_metrics["policy/policy_grad_norm"] = pg_grad_norm_stats.mean()
                local_metrics["policy/ratio_avg"] = ratio_stats.mean()
                local_metrics["policy/ratio_std"] = ratio_stats.std() if ratio_stats.shape[0] > 1 else torch.tensor(0, device=device)
                local_metrics["loss/policy_avg"] = pg_loss_stats.mean()
                local_metrics["loss/value_avg"] = vf_loss_stats.mean()
                if args.use_value_model:
                    local_metrics["value/value_grad_norm"] = vf_grad_norm_stats.mean()
                    local_metrics["value/clipfrac_avg"] = vf_clipfrac_stats.mean()

            # Convert metrics to tensors for reduction
            metric_keys = list(local_metrics.keys())
            metric_values = torch.tensor([local_metrics[k].item() for k in metric_keys if local_metrics[k] is not None], device=device)
            metric_values /= dist.get_world_size()
            dist.all_reduce(metric_values, op=dist.ReduceOp.SUM)
            global_metrics = {k: v.item() for k, v in zip(metric_keys, metric_values)}
            global_metrics.update(
                {
                    "objective/episodic_return": sum(episodic_returns)/len(episodic_returns) if len(episodic_returns) > 0 else 0,
                    "objective/episodic_length": sum(episodic_lengths)/len(episodic_lengths) if len(episodic_lengths) > 0 else 0,
                }
            )
            # Update metrics dictionary
            metrics = {
                "episode": episode,
                "training_step": training_step,
                "lr": self.policy_scheduler.get_last_lr()[0],
                **global_metrics,  # Unpack all the reduced metrics
                **timer.get_log(),  # Unpack all the timing logs
            }

            if accelerator.is_main_process:
                print_rich_single_line_metrics(metrics)
                metrics_queue.put((metrics, global_step))
            del (global_metrics, metrics)
            gc.collect()
            torch.cuda.empty_cache()
            log_gpu_memory_usage("[Training] After training", rank=accelerator.process_index, logger=logger, level=logging.INFO)

            # save steps
            if args.save_freq > 0 and training_step % args.save_freq == 0:
                checkpoint_dir = args.exp_dir
                os.makedirs(checkpoint_dir, exist_ok=True)
                step_dir = os.path.join(checkpoint_dir, f"step_{training_step}")
                os.makedirs(step_dir, exist_ok=True)
                print(f"Saving model at step {training_step} to {step_dir}")
                self.save_model(self.model, processor, step_dir)
        
        print(f"Saving final model at step {training_step} to {args.exp_dir}")
        self.save_model(self.model, processor, args.exp_dir)
        print("finished training")

    def save_model(self, model_to_save: PreTrainedModel, processor: AutoProcessor, output_dir: str) -> None:
        if self._rank == 0:
            os.makedirs(output_dir, exist_ok=True)

            with FSDP.summon_full_params(model_to_save):
                if is_peft_model(model_to_save):
                    model_to_save.save_pretrained(output_dir)
                else:
                    model_state_dict = model_to_save.state_dict()
                    torch.save(model_state_dict, os.path.join(output_dir, 'model.pt'))

                # Save sharded model state
                logger.info(f'Saving model to {os.path.abspath(output_dir)}')

            # Save HF config and tokenizer on rank 0
            hf_path = os.path.join(output_dir, 'huggingface')
            os.makedirs(hf_path, exist_ok=True)
            if hasattr(model_to_save, "config"):
                model_to_save._fsdp_wrapped_module.config.save_pretrained(hf_path)

            # Save the processor
            processor.save_pretrained(output_dir)

        dist.barrier()

def kill_ray_cluster_if_a_worker_dies(object_refs: List[Any], stop_event: threading.Event):
    while True:
        if stop_event.is_set():
            break
        for ref in object_refs:
            try:
                ray.get(ref, timeout=0.01)
            except ray.exceptions.GetTimeoutError:
                pass
            except Exception as e:
                print(e)
                print(f"Actor {ref} died")
                time.sleep(120)
                ray.shutdown()
                os._exit(1)  # Force shutdown the process

        time.sleep(30)


class ModelGroup:
    def __init__(
        self,
        pg: PlacementGroup,
        ray_process_cls: RayProcess,
        num_gpus_per_node: List[int],
    ):
        self.pg = pg
        self.ray_process_cls = ray_process_cls
        self.num_gpus_per_node = num_gpus_per_node
        self.num_gpus_per_actor = 1
        self.num_cpus_per_actor = 4
        self.models = []
        world_size = sum(self.num_gpus_per_node)
        master_policy = ray_process_cls.options(
            num_cpus=self.num_cpus_per_actor,
            num_gpus=self.num_gpus_per_actor,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=self.pg, placement_group_bundle_index=0
            ),
        ).remote(world_size, 0, None, None)

        self.models.append(master_policy)
        master_addr, master_port = ray.get(master_policy.get_master_addr_port.remote())

        def get_bundle_index(rank, num_gpus_per_node):
            """given a rank and a list of num_gpus_per_node, return the index of the bundle that the rank belongs to"""
            bundle_idx = 0
            while rank >= num_gpus_per_node[bundle_idx]:
                rank -= num_gpus_per_node[bundle_idx]
                bundle_idx += 1
            return bundle_idx

        # Setup worker models
        for rank in range(1, world_size):
            print(f"{rank=}, {world_size=}, {master_addr=}, {master_port=}")
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=self.pg,
                placement_group_bundle_index=get_bundle_index(rank, self.num_gpus_per_node),
            )
            worker_policy = ray_process_cls.options(
                num_cpus=self.num_cpus_per_actor,
                num_gpus=self.num_gpus_per_actor,
                scheduling_strategy=scheduling_strategy,
            ).remote(world_size, rank, master_addr, master_port)
            self.models.append(worker_policy)


@draccus.wrap()
def main(args: Args):
    print(f"PPO Fine-tuning OpenVLA Model `{args.vla_path}` on `{args.dataset_name}`")

    calculate_runtime_args(args)

    # Start =>> Build Directories
    run_dir, adapter_dir = args.run_root_dir / args.exp_id, args.adapter_tmp_dir / args.exp_id
    os.makedirs(run_dir, exist_ok=True)

    args.exp_dir = adapter_dir if args.load_adapter_checkpoint is not None else run_dir
    os.makedirs(args.exp_dir, exist_ok=True)
    video_dir = os.path.join(args.exp_dir, "rollouts")
    cprint(f"Clearing existing videos in {video_dir}", "red")
    if os.path.exists(video_dir):
        for f in os.listdir(video_dir):
            if f.endswith(".mp4"):
                os.remove(os.path.join(video_dir, f))

    set_seed_everywhere(args.seed)

    all_configs = {}
    all_configs.update(**asdict(args))
    if args.use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=all_configs,
            name=args.exp_id,
            save_code=True,
            mode="offline" if args.wandb_offline else "online",
            # tags=[args.exp_name] + get_wandb_tags(),
        )
    writer = SummaryWriter(f"logs/tensorboard/{args.exp_id}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # [OpenVLA] Get Hugging Face processor
    processor = None
    if args.model_family == "openvla":
        processor = get_processor(args)
    cprint(f"Loaded processor from {args.vla_path}", "green")

    pg = None
    bundles = [{"GPU": actor_num_gpus, "CPU": actor_num_gpus * 10} for actor_num_gpus in args.actor_num_gpus_per_node]
    pg = placement_group(bundles, strategy="STRICT_SPREAD")
    ray.get(pg.ready())

    inits = []
    policy_group = ModelGroup(
        pg,
        PolicyTrainerRayProcess,
        args.actor_num_gpus_per_node,
    )
    inits.extend(
        model.from_pretrained.remote(args) for model in policy_group.models
    )

    metrics_queue = RayQueue()
    ray.get(inits)

    # [VLLM] Initialize VLLM engines
    max_image_tokens = ray.get(policy_group.models[0].get_max_image_tokens.remote())
    max_len = max_image_tokens + args.context_length + args.response_length
    vllm_engines = create_vllm_engines(
        num_engines=args.vllm_num_engines,
        tensor_parallel_size=args.vllm_tensor_parallel_size,
        enforce_eager=args.vllm_enforce_eager,
        pretrain=args.vla_path,
        revision=None,
        seed=args.seed,
        enable_prefix_caching=args.enable_prefix_caching,
        max_model_len=max_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    # vllm_engines = None

    print("======== all models initialized =========")

    # Save dataset statistics for inference
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder if "v01" not in args.vla_path else VicunaV15ChatPromptBuilder,
    )
    logger.info(f"processor.image_processor.input_sizes: {processor.image_processor.input_sizes}")
    vla_dataset = RLDSDataset(
        args.data_root_dir,
        args.dataset_name,
        batch_transform,
        resize_resolution=tuple(processor.image_processor.input_sizes[0][1:]),
        shuffle_buffer_size=args.shuffle_buffer_size,
        image_aug=args.image_aug,
    )
    save_dataset_statistics(vla_dataset.dataset_statistics, args.exp_dir)

    print("======== all datasets initialized =========")

    refs = []
    for i, policy_model in enumerate(policy_group.models):
        refs.append(
            policy_model.train.remote(
                processor=processor,
                vllm_engines=vllm_engines,
                metrics_queue=metrics_queue,
            )
        )

    # somtimes a worker dies due to CUDA issues, but the rest of the cluster would just hang
    # so we need kill the ray cluster when this happens.
    stop_event = threading.Event()
    threading.Thread(target=kill_ray_cluster_if_a_worker_dies, args=(refs, stop_event)).start()

    # train and gather metrics
    resume_training_step = 1
    for training_step in range(resume_training_step, args.num_training_steps):
        result = metrics_queue.get()
        metrics, global_step = result
        for key, value in metrics.items():
            writer.add_scalar(key, value, global_step=global_step)

    ray.get(refs)

    # save model
    ray.shutdown()
    stop_event.set()

    if args.push_to_hub:
        print("Pushing model to hub")
        # TODO: push to hub


if __name__ == "__main__":
    main()
    print("RL Done!")
