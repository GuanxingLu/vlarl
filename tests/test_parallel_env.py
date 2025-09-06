#!/usr/bin/env python3
"""
Test script for VLAEnv initialization without model setup.
This script mimics the structure of ppo_vllm_ray_fsdp_v3.py
but focuses only on testing the environment initialization.
"""

import os
import sys
import numpy as np
import logging
import random
import socket
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Any
from termcolor import cprint
from copy import deepcopy
import draccus

import ray
from ray.util.placement_group import PlacementGroup, placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

# Add current directory and parent directories to Python path
sys.path.append(".")
sys.path.append("..")

from ppo.envs.libero_env import VLAEnv
from ppo.utils.logging_utils import init_logger
from ppo.utils.ray_utils import ray_noset_visible_devices
from experiments.robot.robot_utils import exact_div

logger = init_logger(__name__)

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set default CUDA devices for the test
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

@dataclass
class TestArgs:
    """Test configuration for testing VLAEnv initialization - mirrors Args from main file"""
    # Common args
    seed: int = 1
    """Seed of the experiment"""

    # Model family (required by VLAEnv)
    model_family: str = "openvla"
    """Model family"""
    
    # Directory Paths
    data_root_dir: Path = Path("datasets/open-x-embodiment")
    """Path to Open-X dataset directory"""
    dataset_name: str = "droid_wipe"
    """Name of fine-tuning dataset (e.g., `droid_wipe`)"""
    run_root_dir: Path = Path("test_runs")
    """Path to directory to store test logs"""
    adapter_tmp_dir: Path = Path("test_adapter-tmp")
    """Temporary directory for test"""

    # Environment Parameters
    task_suite_name: str = "libero_spatial"
    """Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90"""
    num_steps_wait: int = 5  # Reduced for testing
    """Number of steps to wait for objects to stabilize in sim"""
    num_tasks_per_suite: int = 2  # Reduced for testing
    """Number of tasks per suite"""
    num_trials_per_task: int = 2  # Reduced for testing
    """Number of rollouts per task"""
    n_rollout_threads: int = 2  # Reduced for testing
    """Number of parallel vec environments"""
    task_ids: Optional[List[int]] = None
    """Task ids to run"""
    max_env_length: int = 50  # Reduced for testing
    """Max environment length for testing"""
    env_gpu_id: int = 0
    """GPU id for the vectorized environments"""
    context_length: int = 64
    """Length of the query"""

    # Curriculum learning parameters
    use_curriculum: bool = False
    """Whether to use curriculum learning for sampling tasks"""
    curriculum_temp: float = 1.0
    """Temperature parameter for curriculum sampling (higher = more uniform)"""
    curriculum_min_prob: float = 0.0
    """Minimum probability for sampling any task/state"""
    success_history_window: int = 5
    """Number of recent episodes to use for computing success rate"""
    curriculum_recompute_freq: int = 10
    """How often to recompute and log curriculum statistics (in episodes)"""
    
    # For debugging
    verbose: bool = True
    """Whether to print a lot of information"""
    debug: bool = True
    """Whether to run in debug mode"""
    save_video: bool = False  # Disabled for testing
    """Save video of evaluation rollouts"""

    # Augmentation
    image_aug: bool = False  # Disabled for testing
    """Whether to train with image augmentations"""
    center_crop: bool = True
    """Center crop (if trained w/ random crop image aug)"""

    # Reward configuration (required by VLAEnv)
    penalty_reward_value: float = -1.0
    """the reward value for penalty responses"""
    non_stop_penalty: bool = False
    """whether to penalize responses that do not contain stop_token_id"""

    # PPO specific args (required by VLAEnv)
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""

    # Runtime calculated fields
    local_rollout_batch_size: int = 2  # Will be set in calculate_runtime_args
    """The number of rollout episodes per iteration per device"""
    exp_id: str = ""
    """Experiment ID"""
    exp_dir: str = ""
    """Experiment directory"""
    world_size: int = 2 # Default to 2 GPUs based on CUDA_VISIBLE_DEVICES
    """Number of parallel processes"""
    actor_num_gpus_per_node: List[int] = field(default_factory=lambda: [1, 1])
    """number of gpus per node for actor learner"""
    unnorm_key: str = ""
    """unnormalization key"""

def calculate_test_runtime_args(args: TestArgs):
    """Calculate runtime args for testing (mirrors calculate_runtime_args from main file)"""
    args.local_rollout_batch_size = args.n_rollout_threads
    args.world_size = sum(args.actor_num_gpus_per_node)
    
    if args.task_ids is None:
        # For testing, use fewer tasks and allow different batch sizes
        total_actors = sum(args.actor_num_gpus_per_node)
        total_tasks = args.local_rollout_batch_size * total_actors
        args.task_ids = np.array([i % args.num_tasks_per_suite for i in range(total_tasks)])
        logger.info(f"[Test Args] Generated task_ids: {args.task_ids}")

    exp_id = f"test_env+{args.task_suite_name}+tasks{len(np.unique(args.task_ids))}"
    args.exp_id = exp_id
    args.unnorm_key = args.task_suite_name
    cprint(f"[Test Args] Experiment ID: {exp_id}", "green")

class RayProcess:
    """Mirrors RayProcess from main file"""
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

@ray.remote(num_gpus=1)
class VLAEnvTester(RayProcess):
    """A Ray actor to test VLAEnv initialization in parallel."""
    def run_tests(self, args: TestArgs):
        """Initializes VLAEnv and runs basic tests."""
        # Update logger with rank information
        global logger
        logger = init_logger(__name__, self._rank)
        
        actor_args = deepcopy(args)
        local_rollout_indices = slice(self._rank * actor_args.local_rollout_batch_size, (self._rank + 1) * actor_args.local_rollout_batch_size)
        actor_args.task_ids = actor_args.task_ids[local_rollout_indices]
        
        logger.info(f"[Rank {self._rank}] Initializing VLAEnv with task_ids: {actor_args.task_ids}")
        
        try:
            # Initialize the environment
            train_env = VLAEnv(cfg=actor_args, mode="train")
            logger.info(f"[Rank {self._rank}] VLAEnv initialized successfully")
            
            # Test environment reset
            _, info = train_env.reset()
            logger.info(f"[Rank {self._rank}] Environment reset successful, info: {type(info)}")
            
            # Close the environment
            train_env.close()
            logger.info(f"[Rank {self._rank}] Environment closed successfully")
            
            return True
        except Exception as e:
            logger.error(f"[Rank {self._rank}] VLAEnv test failed: {e}")
            raise e

class ModelGroup:
    """A helper class to manage the creation of Ray actors in a placement group - mirrors ModelGroup from main file."""
    def __init__(
        self,
        pg: PlacementGroup,
        ray_process_cls: Any,
        num_gpus_per_node: List[int],
    ):
        self.pg = pg
        self.ray_process_cls = ray_process_cls
        self.num_gpus_per_node = num_gpus_per_node
        self.num_gpus_per_actor = 1
        self.num_cpus_per_actor = 4
        self.actors = []
        world_size = sum(self.num_gpus_per_node)
        
        # Create master actor
        master_actor = ray_process_cls.options(
            num_cpus=self.num_cpus_per_actor,
            num_gpus=self.num_gpus_per_actor,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=self.pg, placement_group_bundle_index=0
            ),
        ).remote(world_size, 0, None, None)

        self.actors.append(master_actor)
        master_addr, master_port = ray.get(master_actor.get_master_addr_port.remote())
        cprint(f"✓ Master actor created at {master_addr}:{master_port}", "green")

        def get_bundle_index(rank, num_gpus_per_node):
            """given a rank and a list of num_gpus_per_node, return the index of the bundle that the rank belongs to"""
            bundle_idx = 0
            while rank >= num_gpus_per_node[bundle_idx]:
                rank -= num_gpus_per_node[bundle_idx]
                bundle_idx += 1
            return bundle_idx

        # Setup worker actors
        for rank in range(1, world_size):
            logger.info(f"{rank=}, {world_size=}, {master_addr=}, {master_port=}")
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=self.pg,
                placement_group_bundle_index=get_bundle_index(rank, self.num_gpus_per_node),
            )
            worker_actor = ray_process_cls.options(
                num_cpus=self.num_cpus_per_actor,
                num_gpus=self.num_gpus_per_actor,
                scheduling_strategy=scheduling_strategy,
            ).remote(world_size, rank, master_addr, master_port)
            self.actors.append(worker_actor)
        
        cprint(f"✓ {len(self.actors)} VLAEnvTester actors created", "green")

def kill_ray_cluster_if_a_worker_dies(object_refs: List[Any], stop_event: threading.Event):
    """Monitors Ray object references and terminates the cluster if a worker fails - mirrors function from main file."""
    while True:
        if stop_event.is_set():
            break
        for ref in object_refs:
            try:
                ray.get(ref, timeout=0.01)
            except ray.exceptions.GetTimeoutError:
                pass
            except Exception as e:
                logger.info(e)
                logger.info(f"Actor {ref} died")
                cprint(f"A worker died, killing the ray cluster: {e}", "red")
                time.sleep(5)  # Reduced sleep time for testing
                ray.shutdown()
                os._exit(1)  # Force shutdown the process
        time.sleep(10)  # Reduced sleep time for testing

@draccus.wrap()
def main(args: TestArgs):
    """Main test function that mirrors the structure of ppo_vllm_ray_fsdp_v3.py but only tests VLAEnv"""
    cprint(f"Starting VLAEnv initialization test for `{args.task_suite_name}`", "blue")
    calculate_test_runtime_args(args)

    # Build Directories (mirrors main file)
    run_dir = args.run_root_dir / args.exp_id
    args.exp_dir = str(run_dir)
    # Don't create directories for testing to avoid file system pollution
    # os.makedirs(args.exp_dir, exist_ok=True)

    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init()

    pg = None
    monitor_thread = None
    stop_event = threading.Event()
    
    try:
        # Create placement group (mirrors main file)
        bundles = [{"GPU": num_gpus, "CPU": num_gpus * 4} for num_gpus in args.actor_num_gpus_per_node]
        pg = placement_group(bundles, strategy="STRICT_SPREAD")
        ray.get(pg.ready())
        cprint("✓ Placement group ready", "green")

        # Create actor group (mirrors main file structure)
        actor_group = ModelGroup(pg, VLAEnvTester, args.actor_num_gpus_per_node)

        cprint("Starting parallel VLAEnv tests...", "blue")
        
        # Start tests on all actors (mirrors main file pattern)
        results_ref = [actor.run_tests.remote(args) for actor in actor_group.actors]
        
        # Start monitoring thread (mirrors main file)
        monitor_thread = threading.Thread(
            target=kill_ray_cluster_if_a_worker_dies, 
            args=(results_ref, stop_event), 
            daemon=True
        )
        monitor_thread.start()

        # Wait for all tests to complete (mirrors main file pattern)
        results = ray.get(results_ref)
        cprint(f"✓ All {len(results)} actors completed VLAEnv tests successfully.", "green")
        assert all(results), "One or more VLAEnv tests failed."

    finally:
        # Cleanup (mirrors main file)
        stop_event.set()
        if monitor_thread:
            monitor_thread.join(timeout=2)
        if pg:
            remove_placement_group(pg)
            cprint("✓ Placement group removed.", "green")
        if ray.is_initialized():
            ray.shutdown()
            cprint("✓ Ray cluster shut down.", "green")

if __name__ == "__main__":
    main()
    logger.info("VLAEnv Test Done!")
