import os
import sys
import numpy as np
import random
from typing import List, TypedDict, Tuple, Dict, Any, Union
from termcolor import cprint
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from ppo.envs.venv import SubprocVectorEnv
from ppo.utils.util import add_info_board
from ppo.envs.base import BaseEnv, EnvOutput


# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import preprocess_input, preprocess_input_batch
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_image_resize_size,
    invert_gripper_action,
    normalize_gripper_action,
)


class VLAEnv(BaseEnv[EnvOutput, np.ndarray]):
    """
    Trajectory-level Reinforcement Learning environment for OpenVLA models in LIBERO tasks.
    """

    def __init__(self, cfg, mode="train"):
        """
        Initialize the environment.

        Args:
            cfg: Configuration object containing model and environment settings.
            mode (str): Mode of the environment, either "train" or "eval".
        """
        super().__init__(seed=cfg.seed)
        self.env_gpu_id = getattr(cfg, "env_gpu_id", 0)

        selected_device = (
            os.environ.get("CUDA_VISIBLE_DEVICES", None)
            if os.environ.get("MUJOCO_EGL_DEVICE_ID", None) is None
            else os.environ.get("MUJOCO_EGL_DEVICE_ID", None)
        )
        cprint(f"[DEBUG] Selected device: {selected_device}", "yellow")
        if mode == "eval" and selected_device is not None:  # TODO: remove this hack
            self.gpu_ids = [int(device) for device in selected_device.split(",")]
            self.gpu_ids = [self.gpu_ids[self.env_gpu_id]]
        else:
            self.gpu_ids = [self.env_gpu_id]
        cprint(f"[DEBUG] GPU IDs: {self.gpu_ids}", "yellow")

        self.num_tasks_per_suite = cfg.num_tasks_per_suite
        self.env_num = min(cfg.n_rollout_threads, self.num_tasks_per_suite)
        self.task_ids = getattr(cfg, "task_ids", None)
        self.max_env_length = cfg.max_env_length
        self.num_steps_wait = cfg.num_steps_wait
        self.model_family = cfg.model_family
        self.save_video = cfg.save_video
        self.center_crop = cfg.center_crop
        self.exp_dir = cfg.exp_dir
        self.cfg = cfg
        self.mode = mode
        assert mode in ["train", "eval"], f"Invalid mode: {mode}"
        self.n_agents = 1  # Number of agents (default: 1) NOT USED

        # Initialize LIBERO task suite
        self.benchmark_dict = benchmark.get_benchmark_dict()
        self.task_suite_name = cfg.task_suite_name
        self.task_suite = self.benchmark_dict[self.task_suite_name]()
        self.num_tasks_in_suite = min(self.task_suite.n_tasks, self.num_tasks_per_suite)    # 10
        self.num_trials_per_task = cfg.num_trials_per_task
        cprint(f"[DEBUG] Task suite: {self.task_suite_name}, Total tasks: {self.num_tasks_in_suite}", "yellow")
        cprint(f"[DEBUG] Task ids: {self.task_ids}", "yellow")

        if self.task_ids is not None:
            assert len(self.task_ids) == self.env_num, f"Number of task ids ({len(self.task_ids)}) must match number of environments ({self.env_num})"

        # Dynamic max_step settings based on task suite
        self.max_step = self._get_max_step(self.task_suite_name)
        cprint(f"[DEBUG] Max steps for task suite '{self.task_suite_name}': {self.max_step}", "yellow")

        # Image resize dimensions
        self.resize_size = get_image_resize_size(self.cfg)

        # Initialize environment variables
        self.env = None
        self.task_descriptions = None
        self.initial_states = None
        self.current_lang_state = None
        self.success = None
        self.total_episodes = 0
        self.last_obs_np_list = None

        # Reward function design
        self.penalty_reward_value = cfg.penalty_reward_value
        self.non_stop_penalty = cfg.non_stop_penalty
        self.verify_reward_value = cfg.verify_reward_value

        # Curriculum learning parameters
        self.use_curriculum = getattr(cfg, "use_curriculum", False)
        self.curriculum_temp = getattr(cfg, "curriculum_temp", 1.0)
        self.curriculum_min_prob = getattr(cfg, "curriculum_min_prob", 0.1)
        self.success_tracker = {}  # {task_id: {state_id: [success_history]}}
        self.success_history_window = getattr(cfg, "success_history_window", 5)
        self.curriculum_recompute_freq = getattr(cfg, "curriculum_recompute_freq", 10)
        self.episode_count = 0

    def _get_max_step(self, task_suite_name: str) -> int:
        """
        Determine max_step dynamically based on the task suite.

        Args:
            task_suite_name (str): Name of the task suite.

        Returns:
            int: Maximum number of steps allowed for the task.
        """
        if self.max_env_length > 0:
            # for debugging
            cprint(f"[Warning] Using max_step from config: {self.max_env_length}", "red")
            return self.max_env_length
        task_max_steps = {
            "libero_spatial": 220,
            "libero_object": 280,
            "libero_goal": 300,
            "libero_10": 520,
            "libero_90": 400,
        }
        return task_max_steps.get(task_suite_name, 300)  # Default to 300 if not specified

    def reset(self) -> Tuple[EnvOutput, Dict[str, Any]]:
        """
        Reset the environment for a new episode.

        Returns:
            Tuple[EnvOutput, Dict[str, Any]]: Initial observation and info dictionary
        """
        obs = self._reset_impl()
        info = {
            "task_description": self.task_descriptions,
            "step_count": self.step_count,
        }
        return obs, info

    def step(self, action: np.ndarray, **kwargs) -> Tuple[EnvOutput, float, bool, Dict[str, Any]]:
        return self._step_impl(action, **kwargs)

    @property
    def action_space(self) -> Tuple[int, ...]:
        return (7,)
    
    @property
    def observation_space(self) -> Dict[str, Tuple[int, ...]]:
        return {
            "pixel_values": (3, 224, 224),
            "prompts": (1,),
        }

    def _get_success_rate(self, task_id, state_id):
        """
        Calculate success rate for a task-state pair.
        
        Args:
            task_id: ID of the task
            state_id: ID of the initial state
            
        Returns:
            float: Success rate (0.0 to 1.0)
        """
        if task_id not in self.success_tracker or state_id not in self.success_tracker[task_id]:
            return 0.0
        
        history = self.success_tracker[task_id][state_id]
        if not history:
            return 0.0
        
        return sum(history) / len(history)

    def _update_success_tracker(self, task_id, state_id, success):
        """
        Update success tracker with the latest result.
        
        Args:
            task_id: ID of the task
            state_id: ID of the initial state
            success: Whether the task was successful (True/False)
        """
        if task_id not in self.success_tracker:
            self.success_tracker[task_id] = {}
        
        if state_id not in self.success_tracker[task_id]:
            self.success_tracker[task_id][state_id] = []
            
        history = self.success_tracker[task_id][state_id]
        history.append(1.0 if success else 0.0)
        
        # Maintain fixed window size
        if len(history) > self.success_history_window:
            history.pop(0)

    def _sample_task_state_curriculum(self, task_id):
        """
        Sample an initial state for a task using curriculum learning.
        
        Args:
            task_id: ID of the task
            
        Returns:
            int: Selected state ID
        """
        task_initial_states = self.initial_states_list[task_id]
        state_ids = list(range(len(task_initial_states)))
        
        # If task not in tracker yet, use uniform sampling
        if task_id not in self.success_tracker:
            return random.choice(state_ids)
        
        # Calculate sampling weights (lower success rate = higher weight)
        weights = []
        for state_id in state_ids:
            success_rate = self._get_success_rate(task_id, state_id)
            
            # Weight inversely proportional to success rate, with temperature
            weight = (1.0 - success_rate) ** (1.0 / self.curriculum_temp)
            
            # Ensure minimum sampling probability
            weight = max(weight, self.curriculum_min_prob)
            weights.append(weight)
        
        # Normalize weights to probabilities
        total_weight = sum(weights)
        probabilities = [w / total_weight for w in weights]
        
        # Sample based on probabilities
        return random.choices(state_ids, weights=probabilities, k=1)[0]

    def get_curriculum_stats(self):
        """
        Get curriculum statistics for logging.
        
        Returns:
            dict: Statistics about the curriculum state
        """
        stats = {}
        for task_id in self.success_tracker:
            for state_id in self.success_tracker[task_id]:
                success_rate = self._get_success_rate(task_id, state_id)
                stats[f"task_{task_id}_state_{state_id}_success"] = success_rate
        
        return stats

    def _reset_impl(self) -> EnvOutput:
        # HACK: 10 tasks in total, so we setup >10 environments
        # as libero does not support multi-tasking within the vec env
        # to make sure we can cover all tasks in the suite

        # Sample tasks (one environment per task)
        if self.env_num < self.num_tasks_in_suite:
            cprint(f"[Warning] {self.env_num} environments are being used for {self.num_tasks_in_suite} tasks.", "red")
        
        if self.mode == "train":
            if self.task_ids is None:
                task_ids = random.sample(range(self.num_tasks_in_suite), self.env_num)
            else:
                task_ids = self.task_ids
        elif self.mode == "eval":
            task_ids = list(range(min(self.num_tasks_in_suite, self.env_num)))
            # task_ids = [1]

        cprint(f"[DEBUG] Sampled task_ids: {task_ids}", "yellow")

        self.tasks = []
        self.initial_states_list = []
        env_creators = []
        resolution = 256

        # Prepare environments and collect task information
        for id, task_id in enumerate(task_ids):
            task = self.task_suite.get_task(task_id)
            self.tasks.append(task)
            task_initial_states = self.task_suite.get_task_init_states(task_id)

            if len(task_initial_states) > self.num_trials_per_task:
                cprint(f"[Warning] Task {task_id} has {len(task_initial_states)} initial states. Truncating to {self.num_trials_per_task}.", "red")
                task_initial_states = task_initial_states[:self.num_trials_per_task]
                
            self.initial_states_list.append(task_initial_states)
            
            task_bddl_file = os.path.join(get_libero_path("bddl_files"), 
                                        task.problem_folder, 
                                        task.bddl_file)
            
            # Distribute environments evenly across available GPUs
            assigned_gpu = self.gpu_ids[id % len(self.gpu_ids)]
            
            env_args = {
                "bddl_file_name": task_bddl_file,
                "camera_heights": resolution,
                "camera_widths": resolution,
                "render_gpu_device_id": assigned_gpu,  # Assign specific GPU
            }
            # cprint(f"[DEBUG] Env {id} assigned to GPU {assigned_gpu}", "yellow")
            # cprint(f"[DEBUG] Env args: {env_args}", "yellow")
            env_creators.append(lambda args=env_args: OffScreenRenderEnv(**args))

        # Create parallel environments
        self.env = SubprocVectorEnv(env_creators)
        self.env.seed(self.seed)
        self.env.reset()

        if self.mode == "eval":
            self.to_do_state_ids = {}
            for task_idx in range(self.env_num):
                self.to_do_state_ids[task_idx] = list(range(len(self.initial_states_list[task_idx])))

        # Sample one initial state for each task
        self.initial_state_ids = []
        initial_states_to_set = []
        for task_idx in range(self.env_num):
            task_initial_states = self.initial_states_list[task_idx]
            if self.mode == "train":
                if self.use_curriculum:
                    state_id = self._sample_task_state_curriculum(task_idx)
                else:
                    state_id = random.randint(0, len(task_initial_states) - 1)
            elif self.mode == "eval":
                state_id = self.to_do_state_ids[task_idx].pop(0)

            self.initial_state_ids.append(state_id)
            initial_states_to_set.append(task_initial_states[state_id])

        cprint(f"[DEBUG] Initial state ids: {self.initial_state_ids}", "yellow")

        # Set initial states for all environments
        obs = self.env.set_init_state(initial_states_to_set)

        # Store task descriptions
        self.task_descriptions = [task.language for task in self.tasks]

        # Wait for stabilization
        for _ in range(self.num_steps_wait):
            dummy_action = get_libero_dummy_action(self.model_family)
            dummy_actions = [dummy_action for _ in range(self.env_num)]
            obs, _, _, _ = self.env.step(dummy_actions)

        obs_np_list = obs

        # Prepare initial observation
        img_list = [get_libero_image(obs, self.resize_size) for obs in obs_np_list]
        prompt_list = self.task_descriptions
        img_list, prompt_list = preprocess_input_batch(img_list, prompt_list, pre_thought_list=None, center_crop=self.center_crop)

        self.current_lang_state = prompt_list
        self.step_count = np.zeros(self.env_num)
        self.success = np.zeros(self.env_num, dtype=bool)

        self.last_obs_np_list = obs_np_list
        self.replay_images = {task_idx: [] for task_idx in range(self.env_num)}
        # self.replay_values = {task_idx: [] for task_idx in range(self.env_num)}

        if self.save_video: # we only save video for the first environmen
            self.save_dir = os.path.join(self.exp_dir, "rollouts")
            os.makedirs(self.save_dir, exist_ok=True)
            for task_idx, obs in enumerate(obs_np_list):
                img = get_libero_image(obs, self.resize_size)
                img_args = {
                    "goal": self.task_descriptions[task_idx],
                }
                img = add_info_board(img, **img_args)
                self.replay_images[task_idx].append(img)

        return EnvOutput(pixel_values=img_list, prompts=prompt_list)

    def _step_impl(self, action: np.ndarray, **kwargs) -> Tuple[EnvOutput, np.ndarray, np.ndarray, Dict]:
        self.step_count += 1

        step_count_tmp = self.step_count.copy()
        
        # responses not passing that filter will receive a low (fixed) score
        invalid_ids = []
        if self.mode == "train" and self.non_stop_penalty:
            invalid_mask = (action == -100.0).any(axis=1)
            if invalid_mask.any():
                cprint(f"[Warning] Invalid action: {action}. Penalizing the agent.", "red")
                dummy_action = get_libero_dummy_action(self.model_family)
                action[invalid_mask] = dummy_action
                invalid_ids = np.where(invalid_mask)[0].tolist()

        # Normalize and execute the action
        normalized_action = normalize_gripper_action(action, binarize=True)
        if self.model_family == "openvla":
            normalized_action = invert_gripper_action(normalized_action)
        # Execute the action in the environment
        # valid_ids = [idx for idx in range(self.env_num) if idx not in invalid_ids]
        valid_task_ids = [idx for idx in range(self.env_num) if self.initial_state_ids[idx] != -1]   # skip the task if no more initial state to reset
        id_to_task_id = {idx: task_id for idx, task_id in enumerate(valid_task_ids)}    # map from valid index (e.g., 0, 1, 2) to task_id (e.g., 2, 5, 10)

        obs_np_list, reward_np_list, done_np_list, info = self.env.step(normalized_action.tolist(), id=valid_task_ids)
        # obs_np_list: [len(valid_ids), 256, 256, 3]
        # ['robot0_joint_pos', 'robot0_joint_pos_cos', 'robot0_joint_pos_sin', 'robot0_joint_vel', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'robot0_gripper_qvel', 'agentview_image', 'robot0_eye_in_hand_image', 'porcelain_mug_1_pos', 'porcelain_mug_1_quat', 'porcelain_mug_1_to_robot0_eef_pos', 'porcelain_mug_1_to_robot0_eef_quat', 'red_coffee_mug_1_pos', 'red_coffee_mug_1_quat', 'red_coffee_mug_1_to_robot0_eef_pos', 'red_coffee_mug_1_to_robot0_eef_quat', 'plate_1_pos', 'plate_1_quat', 'plate_1_to_robot0_eef_pos', 'plate_1_to_robot0_eef_quat', 'chocolate_pudding_1_pos', 'chocolate_pudding_1_quat', 'chocolate_pudding_1_to_robot0_eef_pos', 'chocolate_pudding_1_to_robot0_eef_quat', 'robot0_proprio-state', 'object-state']

        # Scale the positive rewards
        reward_np_list = np.array(reward_np_list)
        if self.mode == "train":
            # reward_np_list[reward_np_list > 0] = self.verify_reward_value
            # Penalize the invalid actions
            reward_np_list[invalid_ids] = self.penalty_reward_value

        # Determine rewards and done status
        dones = [done or self.step_count[id_to_task_id[idx]] >= self.max_step for idx, done in enumerate(done_np_list)] # (len(valid_ids),)
        dones = np.array(dones)

        # Save the current frame for all environments
        if self.save_video:
            values = kwargs.get("values", None)
            log_probs = kwargs.get("log_probs", None)
            prm_rewards = kwargs.get("prm_rewards", None)
            probs = np.exp(log_probs) if log_probs is not None else None
            for idx in range(len(obs_np_list)):
                img = get_libero_image(obs_np_list[idx], self.resize_size)
                img_args = {
                    # "goal": self.task_descriptions[idx],
                    "step": self.step_count[idx],
                    "tokens": action[idx],
                    "prob": probs[idx] if probs is not None else None,
                    "value_preds": values[idx] if values is not None else None,
                    "prm_rewards": prm_rewards[idx] if prm_rewards is not None else None,
                }
                img = add_info_board(img, **img_args)
                self.replay_images[idx].append(img)
                # if values is not None:
                #     self.replay_values[idx].append(values[idx])

        # Logging & Resetting
        if np.any(dones):
            # Update success and step count for done environments
            done_indices = np.where(dones)[0]
            done_task_indices = [id_to_task_id[idx] for idx in done_indices]  # map back to task_id

            self.success[done_task_indices] = np.array(reward_np_list)[done_indices] > 0
            
            new_initial_states = []
            dummy_actions = []
            for i, task_id in enumerate(done_task_indices):
                # Update success tracker and log results
                done_idx = done_indices[i]
                success = reward_np_list[done_idx] > 0
                
                if self.mode == "train" and self.use_curriculum:
                    # Update success tracker with this episode's result
                    current_state_id = self.initial_state_ids[task_id]
                    self._update_success_tracker(task_id, current_state_id, success)
                
                # Log results
                status = "successfully" if success else "failed to"
                color = "green" if success else "red"
                cprint(f"[INFO] Task {task_id} variant {self.initial_state_ids[task_id]} {status} complete in {self.step_count[task_id]} steps.", color)

                # Save video for done environments if enabled
                if self.save_video:
                    done_idx = done_indices[i]
                    success = reward_np_list[done_idx] > 0
                    processed_task_description = self.task_descriptions[done_idx].lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
                    mp4_path = os.path.join(
                        self.save_dir, f"rank_{self.env_gpu_id}--episode={self.total_episodes + i}--success={success}--task={processed_task_description}.mp4"
                    )
                    save_rollout_video(
                        self.replay_images[done_idx], self.total_episodes + i, success=success, 
                        task_description=self.task_descriptions[done_idx], log_file=None, mp4_path=mp4_path,
                    )
                    self.replay_images[done_idx] = []  # Reset the replay images for this task

                # Sample new states for done environments (keeping the same task)
                task_initial_states = self.initial_states_list[task_id]     # reuse the same task

                if self.mode == "train":
                    if self.use_curriculum:
                        state_id = self._sample_task_state_curriculum(task_id)
                    else:
                        state_id = random.randint(0, len(task_initial_states) - 1)
                elif self.mode == "eval":
                    if len(self.to_do_state_ids[task_id]) == 0:
                        cprint(f"[INFO] Task {task_id} completes evaluation, and has no more initial states to evaluate.", "cyan")
                        self.initial_state_ids[task_id] = -1    # skip this task
                        continue
                    state_id = self.to_do_state_ids[task_id].pop(0)

                self.initial_state_ids[task_id] = state_id
                new_initial_states.append(task_initial_states[state_id])

                dummy_action = get_libero_dummy_action(self.model_family)
                dummy_actions.append(dummy_action)

            self.total_episodes += len(done_task_indices)
            self.episode_count += len(done_task_indices)

            # Reset done environments in parallel
            reset_task_indices = [task_id for task_id in done_task_indices if self.initial_state_ids[task_id] != -1]  # skip the task if no more initial state to reset
            reset_indices = [idx for idx in range(len(dones)) if idx in done_indices and self.initial_state_ids[id_to_task_id[idx]] != -1]
            if len(reset_task_indices) > 0:
                self.env.reset(id=reset_task_indices)
                obs = self.env.set_init_state(init_state=new_initial_states, id=reset_task_indices)
                # Wait for stabilization
                for _ in range(self.num_steps_wait):
                    obs, _, _, _ = self.env.step(action=dummy_actions, id=reset_task_indices)
                    
                # Update observations using mask operation
                obs_np_list[reset_indices] = obs
                # Reset other variables
                self.step_count[reset_task_indices] = 0

            # Filter out completed tasks with no further states
            obs_np_list = [obs_np_list[idx] for idx in range(len(dones)) if idx not in done_indices or self.initial_state_ids[id_to_task_id[idx]] != -1]
            self.task_descriptions = [self.task_descriptions[idx] for idx in range(len(dones)) if idx not in done_indices or self.initial_state_ids[id_to_task_id[idx]] != -1]

        self.last_obs_np_list = obs_np_list

        # Prepare the next observation
        img_list = [get_libero_image(obs, self.resize_size) for obs in obs_np_list]
        prompt_list = self.task_descriptions    # unchanged

        img_list, prompt_list = preprocess_input_batch(img_list, prompt_list, pre_thought_list=None, center_crop=self.center_crop)

        next_obs = EnvOutput(pixel_values=img_list, prompts=prompt_list)

        self.current_lang_state = prompt_list

        infos = {
            "task_description": self.task_descriptions,
            "step_count": self.step_count,
            "step_count_tmp": step_count_tmp,
        }
        
        # Add curriculum stats to info dict periodically
        if self.mode == "train" and self.use_curriculum and self.episode_count >= self.curriculum_recompute_freq:
            self.episode_count = 0
            infos["curriculum_stats"] = self.get_curriculum_stats()

        return next_obs, reward_np_list, dones, infos

    def close(self):
        """Close the environment."""
        self.env.close()
