import os
import time
import numpy as np
# import gymnasium as gym
import gym
import logging
from typing import Dict, Any, Tuple, List, Optional
from PIL import Image
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from envs.base import EnvOutput
from envs.venv import SubprocVectorEnv
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_image,
)
from experiments.robot.openvla_utils import preprocess_input_batch
from experiments.robot.robot_utils import normalize_gripper_action, invert_gripper_action
from utils.logging_utils import init_logger

logger = init_logger(__name__)
logging.getLogger("imageio_ffmpeg").setLevel(logging.ERROR)

class LiberoVecEnv(gym.Env):
    def __init__(
        self,
        task_suite_name: str,
        task_ids: List[int],
        num_trials_per_task: int = 50,
        seed: int = 42,
        model_family: str = "openvla",
        center_crop: bool = True,
        rand_init_state: bool = True,
        num_envs: Optional[int] = None,
        num_steps_wait: Optional[int] = 10,
        max_episode_length: Optional[int] = None,
        resolution: Optional[int] = 256,
        resize_size: Optional[Tuple[int, int]] = None,
        penalty_value: Optional[float] = 0.0,
        state_sampler: Optional[callable] = None,
        auto_reset: bool = True,
        group_size: Optional[int] = None,
        num_group: Optional[int] = None,
    ):
        super().__init__()
        self.task_suite_name = task_suite_name
        self.task_ids = task_ids
        self.group_size = group_size
        self.num_group = num_group
        self.num_envs = num_group * group_size
        self.is_vector_env = True
        self.model_family = model_family
        self.center_crop = center_crop
        self.resize_size = resize_size or (224, 224)
        self.num_trials_per_task = num_trials_per_task
        self.resolution = resolution
        self.seed_ = seed
        self.rand_init_state = rand_init_state
        self.num_steps_wait = num_steps_wait
        self.penalty_value = penalty_value
        self.auto_reset = auto_reset
        self._done_mask = np.zeros(self.num_envs, dtype=bool)
        self._last_obs_list = None
        
        self.benchmark_dict = benchmark.get_benchmark_dict()
        self.task_suite = self.benchmark_dict[self.task_suite_name]()
        self.tasks = []
        self.initial_states_list = []
        self.task_descriptions = []
        if max_episode_length is not None:
            self.max_steps = max_episode_length
        else:
            self.max_steps = self._get_max_step(self.task_suite_name)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Dict({
            "pixel_values": gym.spaces.Box(
                low=0, high=255, shape=(224, 224, 3), dtype=np.uint8
            ),
            "prompts": gym.spaces.Text(max_length=1000)
        })
        
        cuda_visible_devices = os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        gpus = [int(x) for x in cuda_visible_devices.split(",")] if cuda_visible_devices else [0]
        logger.info(f"{gpus=}")

        env_creators = []
        for i in range(self.num_envs):
            task_id = task_ids[i % len(task_ids)]
            task = self.task_suite.get_task(task_id)
            self.tasks.append(task)
            self.task_descriptions.append(task.language)

            task_initial_states = self.task_suite.get_task_init_states(task_id)
            if len(task_initial_states) > self.num_trials_per_task:
                task_initial_states = task_initial_states[:self.num_trials_per_task]
            self.initial_states_list.append(task_initial_states)

            bddl_file = os.path.join(
                get_libero_path("bddl_files"),
                task.problem_folder,
                task.bddl_file
            )
            cur_gpu = gpus[i % len(gpus)]
            env_args = {
                "bddl_file_name": bddl_file,
                "camera_heights": resolution,
                "camera_widths": resolution,
                "render_gpu_device_id": cur_gpu,
                "seed": int(seed),
            }

            def env_fn(param=env_args):
                seed = param.pop("seed")
                env = OffScreenRenderEnv(**param)
                env.seed(seed)
                return env

            env_creators.append(env_fn)

        self.envs = SubprocVectorEnv(env_creators)
        # self.envs.seed(int(self.seed_))
        if cuda_visible_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        self.step_counts = np.zeros(self.num_envs)
        self.current_state_indices = np.zeros(self.num_envs, dtype=int)
        self.task_state_results = {}  # {(task_id, state_id): [success_results]}
        self.current_task_state_pairs = []  # Current (task_id, state_id) for each env
        self.state_sampler = state_sampler  # Sampler hook to choose state_id
        
    def _get_max_step(self, task_suite_name: str) -> int:
        task_max_steps = {
            "libero_spatial": 220,
            "libero_object": 280,
            "libero_goal": 300,
            "libero_10": 520,
            "libero_90": 400,
        }
        return task_max_steps.get(task_suite_name, 300)
        
    def reset(self, seed=None, options=None):
        self.step_counts = np.zeros(self.num_envs)
        self._done_mask[:] = False
        self.task_state_results = {}
        self.current_task_state_pairs = []

        init_states = []

        # Sample state_ids for each group, then repeat for group_size
        group_state_ids = []
        for group_idx in range(self.num_group):
            # Get the environment index for this group (first env in the group)
            env_idx = group_idx * self.group_size
            task_id = self.task_ids[env_idx % len(self.task_ids)]

            if self.state_sampler is not None:
                state_id = int(self.state_sampler(task_id=task_id, n_states=len(self.initial_states_list[env_idx])))
            elif self.rand_init_state:
                state_id = np.random.randint(0, len(self.initial_states_list[env_idx]))
            else:
                state_id = self.current_state_indices[env_idx] % len(self.initial_states_list[env_idx])
                self.current_state_indices[env_idx] += 1

            group_state_ids.append(state_id)

        # Now assign states to all environments based on their group
        for i in range(self.num_envs):
            group_idx = i // self.group_size
            state_id = group_state_ids[group_idx]
            task_id = self.task_ids[i % len(self.task_ids)]

            init_states.append(self.initial_states_list[i][state_id])
            self.current_task_state_pairs.append((task_id, state_id))

        if seed is not None:
            self.envs.seed([seed for i in range(self.num_envs)])
        self.envs.reset()
        obs_list = self.envs.set_init_state(init_states)
        dummy_action = get_libero_dummy_action()
        dummy_actions = [dummy_action] * self.num_envs

        for _ in range(self.num_steps_wait):
            obs_list, _, _, _ = self.envs.step(dummy_actions)

        self._last_obs_list = obs_list

        pixel_values = []
        prompts = []
        for i, obs in enumerate(obs_list):
            img = get_libero_image(obs, self.resize_size)
            if isinstance(img, Image.Image):
                img = np.array(img)
            pixel_values.append(img)
            prompts.append(self.task_descriptions[i])

        img_list, prompt_list = preprocess_input_batch(
            pixel_values, prompts,
            pre_thought_list=None, center_crop=self.center_crop
        )
        env_output = EnvOutput(pixel_values=img_list, prompts=prompt_list)
        info = {
            "task_descriptions": prompts,
            "step_counts": self.step_counts.copy(),
            "penalty_nums": np.array([0] * self.num_envs),
        }
        return env_output, info

    def step(self, actions, **kwargs):
        dummy = np.array(get_libero_dummy_action(), dtype=float)
        is_dummy_mask = np.all(np.isclose(actions, dummy, atol=1e-6), axis=1)

        actions = normalize_gripper_action(actions, binarize=True)
        if self.model_family == "openvla":
            actions = invert_gripper_action(actions)

        if not self.auto_reset:
            active_indices = np.where(~self._done_mask)[0].tolist()

            if len(active_indices) == 0:    # All envs are done
                obs_list = self._last_obs_list
                rewards = np.zeros(self.num_envs, dtype=np.float32)
                dones = np.ones(self.num_envs, dtype=bool)
                is_dummy_mask = np.zeros(self.num_envs, dtype=bool)

                pixel_values = []
                prompts = []
                for i, obs in enumerate(obs_list):
                    img = get_libero_image(obs, self.resize_size)
                    pixel_values.append(img)
                    prompts.append(self.task_descriptions[i])
                img_list, prompt_list = preprocess_input_batch(
                    pixel_values, prompts,
                    pre_thought_list=None, center_crop=True
                )
                env_output = EnvOutput(pixel_values=img_list, prompts=prompt_list)
                info = {
                    "task_descriptions": prompts,
                    "step_counts": self.step_counts.copy(),
                    "penalty_nums": is_dummy_mask,
                }
                truncated = np.array([False] * self.num_envs)
                return env_output, rewards, dones, truncated, info

            active_actions = [actions[i] for i in active_indices]
            active_obs_list, active_rewards, active_dones, active_infos = self.envs.step(
                active_actions, id=active_indices
            )

            obs_list = self._last_obs_list.copy() if self._last_obs_list is not None else [None] * self.num_envs
            rewards = np.zeros(self.num_envs, dtype=np.float32)
            dones = self._done_mask.copy()

            for k, idx in enumerate(active_indices):
                obs_list[idx] = active_obs_list[k]
                rewards[idx] = active_rewards[k]
                dones[idx] = active_dones[k]

            for i in active_indices:
                if is_dummy_mask[i]:
                    rewards[i] += float(self.penalty_value)

            self.step_counts[active_indices] += 1

            for i in active_indices:
                if self.step_counts[i] >= self.max_steps:
                    dones[i] = True

        else:
            obs_list, rewards, dones, infos = self.envs.step(actions)

            for i, is_dummy in enumerate(is_dummy_mask):
                if is_dummy:
                    rewards[i] += float(self.penalty_value)

            self.step_counts += 1

            for i in range(self.num_envs):
                if self.step_counts[i] >= self.max_steps:
                    dones[i] = True

        if np.any(dones):
            done_indices = np.where(dones)[0]
            for i in done_indices:
                task_id, state_id = self.current_task_state_pairs[i]
                success = rewards[i] > 0
                if (task_id, state_id) not in self.task_state_results:
                    self.task_state_results[(task_id, state_id)] = []
                self.task_state_results[(task_id, state_id)].append(success)

            if self.auto_reset:
                new_initial_states = []
                dummy_actions = []

                # Group done indices by their group_idx
                group_state_mapping = {}
                for i in done_indices:
                    group_idx = i // self.group_size
                    if group_idx not in group_state_mapping:
                        # Sample a new state_id for this group
                        env_idx = group_idx * self.group_size
                        task_id = self.task_ids[env_idx % len(self.task_ids)]

                        if self.state_sampler is not None:
                            state_id = int(self.state_sampler(task_id=task_id, n_states=len(self.initial_states_list[env_idx])))
                            state_id = max(0, min(state_id, len(self.initial_states_list[env_idx]) - 1))
                        elif self.rand_init_state:
                            state_id = np.random.randint(0, len(self.initial_states_list[env_idx]))
                        else:
                            state_id = self.current_state_indices[env_idx] % len(self.initial_states_list[env_idx])
                            self.current_state_indices[env_idx] += 1

                        group_state_mapping[group_idx] = state_id

                # Assign states to done environments based on their group
                for i in done_indices:
                    group_idx = i // self.group_size
                    state_id = group_state_mapping[group_idx]
                    task_id = self.task_ids[i % len(self.task_ids)]

                    new_initial_states.append(self.initial_states_list[i][state_id])
                    self.current_task_state_pairs[i] = (task_id, state_id)
                    dummy_action = get_libero_dummy_action()
                    dummy_actions.append(dummy_action)
                self.envs.reset(id=done_indices.tolist())
                obs = self.envs.set_init_state(new_initial_states, id=done_indices.tolist())

                for _ in range(self.num_steps_wait):
                    obs, _, _, _ = self.envs.step(dummy_actions, id=done_indices.tolist())

                for i, done_idx in enumerate(done_indices):
                    obs_list[done_idx] = obs[i]

                for i in done_indices:
                    self.step_counts[i] = 0

        self._done_mask = dones
        self._last_obs_list = obs_list

        pixel_values = []
        prompts = []
        for i, obs in enumerate(obs_list):
            img = get_libero_image(obs, self.resize_size)
            pixel_values.append(img)
            prompts.append(self.task_descriptions[i])
        img_list, prompt_list = preprocess_input_batch(
            pixel_values, prompts,
            pre_thought_list=None, center_crop=True
        )
        env_output = EnvOutput(pixel_values=img_list, prompts=prompt_list)
        info = {
            "task_descriptions": prompts,
            "step_counts": self.step_counts.copy(),
            "penalty_nums": is_dummy_mask,
        }
        truncated = np.array([False] * self.num_envs)

        return env_output, np.array(rewards), np.array(dones), truncated, info

    def is_eval_complete(self) -> bool:
        """Check if all task+initial state pairs have been tested at least once."""
        all_pairs = set()
        for i in range(self.num_envs):
            task_id = self.task_ids[i % len(self.task_ids)]
            for state_id in range(len(self.initial_states_list[i])):
                all_pairs.add((task_id, state_id))
        tested_pairs = set(self.task_state_results.keys())
        return all_pairs.issubset(tested_pairs)
    
    def get_task_state_results(self) -> Dict[Tuple[int, int], List[bool]]:
        return self.task_state_results.copy()

    def get_completed_status(self) -> Tuple[float, int]:
        if not self.task_state_results:
            return 0.0, 0
        total_pairs = len(self.task_state_results)
        successful_pairs = sum(1 for results in self.task_state_results.values() if results[0])
        success_rate = successful_pairs / total_pairs if total_pairs > 0 else 0.0
        per_task_success_rates = {
            f"task_{task_id}": sum(1 for (t_id, _), results in self.task_state_results.items()
                                                if t_id == task_id and results[0]) /
            sum(1 for (t_id, _ ) in self.task_state_results.keys() if t_id == task_id)
            for task_id in set(self.task_ids) if task_id in [pair[0] for pair in self.task_state_results.keys()]
        }
        infos = {
            "success_rate": success_rate,
            "completed_episodes": total_pairs,
            **per_task_success_rates,
        }
        return infos

    def close(self):
        self.envs.close()
    
    def __len__(self):
        return self.num_envs
