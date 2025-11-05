"""
Usage:
    pytest tests/test_ppo.py -s
    pytest tests/test_ppo.py::test_ppo_multi_gpu -s
"""

import subprocess
import os

def test_ppo_multi_gpu():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = os.environ.copy()
    # devices = "0,1,2,3"
    devices = "4,5,6,7"
    # devices = "0,1"
    env.update({
        "MESA_GL_VERSION_OVERRIDE": "4.1",
        "PYOPENGL_PLATFORM": "egl", 
        "MUJOCO_GL": "egl",
        "CUDA_VISIBLE_DEVICES": devices,
    })
    per_device_train_batch_size=4   # ddp
    local_rollout_batch_size=10
    num_gpus = len(devices.split(","))
    actor_gpus = num_gpus - 1
    cmd = [
        f"python",
        "ppo_vllm_ray_fsdp.py",
        "--pretrained_checkpoint", "MODEL/openvla-7b-finetuned-libero-goal",
        "--data_root_dir", "./data/modified_libero_rlds",
        "--dataset_name", "libero_goal_no_noops",
        "--task_suite_name", "libero_goal",
        "--num_trials_per_task", "2",  # Modified for testing
        "--eval_num_trials_per_task", "1",
        "--task_ids", "[0,1,2,3,4,5,6,7,8,9]",  # Modified for testing
        "--run_root_dir", "checkpoints/debug/root",
        "--adapter_tmp_dir", "checkpoints/debug/adapter",
        "--per_device_train_batch_size", f"{per_device_train_batch_size}",
        "--local_mini_batch_size", f"{per_device_train_batch_size}",
        "--local_rollout_batch_size", f"{local_rollout_batch_size}",
        "--local_rollout_forward_batch_size", f"{local_rollout_batch_size}",
        "--actor_num_gpus_per_node", f"[{actor_gpus}]",
        "--temperature", "1.7",
        "--num_epochs", "1",
        "--value_init_steps", "0",  # Modified for testing
        "--learning_rate", "2e-5",
        "--value_learning_rate", "5e-5",
        "--policy_max_grad_norm", "1.0",
        "--value_max_grad_norm", "1.0",
        "--cliprange_high", "0.4",
        "--cliprange_low", "0.2",
        "--gamma", "1.0",
        "--num_steps", "8",  # Modified for testing
        # "--num_steps", "128",
        "--max_env_length", "8",  # Modified for testing
        "--total_episodes", "100000",
        "--vllm_tensor_parallel_size", "1",
        "--vllm_enforce_eager", "True",
        "--enable_prefix_caching", "False",
        # "--vllm_enforce_eager", "False",
        # "--enable_prefix_caching", "True",
        "--gpu_memory_utilization", "0.9",
        "--use_lora", "False",
        "--enable_gradient_checkpointing", "False",
        "--sharding_strategy", "shard-grad-op",
        "--offload", "False",
        "--use_value_model", "True",
        "--value_model_type", "vla",
        "--value_use_lora", "False",
        "--clip_vloss", "False",
        "--norm_adv", "True",
        "--use_curriculum", "True",
        # "--use_curriculum", "False",
        "--curriculum_temp", "1.0",
        "--curriculum_min_prob", "0.0",
        "--save_freq", "10",  # Save more frequently for testing
        "--eval_freq", "10",  # Evaluate more frequently for testing
        # "--init_eval", "True",
        "--init_eval", "False",
        # "--save_video", "True",
        "--save_video", "False",
        # "--use_wandb", "True",
        "--use_wandb", "False",
        "--wandb_offline", "False",
        "--wandb_project", "openvla",
        "--wandb_entity", "openvla_cvpr",
        "--debug", "False"
    ]
    subprocess.run(
        cmd,
        cwd=project_root,
        env=env,
        check=True,
    )
