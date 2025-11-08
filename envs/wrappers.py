from collections import OrderedDict
import os
import time
import shutil
from termcolor import cprint
import numpy as np
# import gymnasium as gym
import gym
import torch
import cv2
from collections import deque
from typing import Dict, List, Optional, Any, Tuple

from utils.plot_utils import add_info_board

DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")


def convert_img(img):
    # Handle PIL Image
    from PIL import Image
    if isinstance(img, Image.Image):
        img = np.array(img)
        if img.dtype != np.uint8:
            img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        return img
    # Handle torch Tensor
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    # Handle numpy array
    if img.dtype != np.uint8:
        img_min = img.min()
        img_max = img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    # Transpose if in CHW format (check if first dimension is smallest, likely channels)
    if len(img.shape) == 3 and img.shape[0] <= 4:
        img = np.transpose(img, (1, 2, 0))
    return img


def save_rollout_video(rollout_images, mp4_path=None, fps=30):
    os.makedirs(os.path.dirname(mp4_path), exist_ok=True)

    if len(rollout_images) == 0:
        return mp4_path

    h, w = rollout_images[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(mp4_path, fourcc, fps, (w, h))

    for frame in rollout_images:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame)

    writer.release()
    print(f"Saved rollout MP4 at path {mp4_path} with {len(rollout_images)} frames")

    return mp4_path

class VideoWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        rank: int = 0,
        save_dir: str = "video",
        save_freq: int = 1,
        save_stats: bool = False,
        max_videos_per_env: int = 1000000,
    ):
        super().__init__(env)
        self.env = env
        self.rank = rank
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.save_stats = save_stats
        self.max_videos_per_env = max_videos_per_env

        self.num_envs = env.num_envs
        self.frames = [[] for _ in range(self.num_envs)]
        self.episode_counts = [0] * self.num_envs
        self.video_counts = [0] * self.num_envs
        self.task_descriptions = env.task_descriptions
        self.replay_images = {i: [] for i in range(self.num_envs)}
        self.total_episodes = 0
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        self.frames = [[] for _ in range(self.num_envs)]
        self.replay_images = {i: [] for i in range(self.num_envs)}

        pixel_values = obs["pixel_values"]
        for i in range(min(len(pixel_values), self.num_envs)):
            if self.video_counts[i] < self.max_videos_per_env:
                img = pixel_values[i]
                img = convert_img(img)
                if self.save_stats:
                    img_args = {
                        "goal": self.task_descriptions[i],
                        "step": self.env.step_counts[i],
                    }
                    img = add_info_board(img, **img_args)
                self.frames[i].append(img)
                self.replay_images[i].append(img)
        
        return obs, info
    
    def step(self, actions, **kwargs):
        values = kwargs.get('values', None)
        log_probs = kwargs.get('log_probs', None)
        prm_rewards = kwargs.get('prm_rewards', None)

        obs, rewards, dones, truncated, info = self.env.step(actions, **kwargs)

        # Handle dones as tensor or numpy array
        if isinstance(dones, torch.Tensor):
            done_np = dones.detach().cpu().numpy().astype(bool)
        else:
            done_np = np.asarray(dones, dtype=bool)

        if np.any(done_np):
            done_indices = np.where(done_np)[0]
            for i in done_indices:
                if self.frames[i] and len(self.frames[i]) > 3 and self.video_counts[i] < self.max_videos_per_env:
                    # Get success from info if available, otherwise use reward
                    if 'episode' in info and 'success' in info['episode']:
                        s = info['episode']['success']
                        if isinstance(s, (list, np.ndarray)):
                            success = bool(s[i])
                        elif isinstance(s, torch.Tensor):
                            success = bool(s[i].item())
                        else:
                            success = bool(s)
                    else:
                        success = rewards[i] > 0
                    self._save_video(i, success)
                    self.video_counts[i] += 1
                    self.total_episodes += 1
                self.episode_counts[i] += 1
                self.frames[i] = []

        pixel_values = obs["pixel_values"]
        for i in range(min(len(pixel_values), self.num_envs)):
            if self.video_counts[i] < self.max_videos_per_env and len(self.frames[i]) < 10000:
                img = pixel_values[i]
                img = convert_img(img)
                if self.save_stats:
                    img_args = {
                        "goal": self.task_descriptions[i],
                        "step": self.env.step_counts[i],
                        "action": actions[i],
                    }
                    if values is not None:
                        img_args["value"] = values[i]
                    if log_probs is not None:
                        lp = log_probs[i]
                        if isinstance(lp, torch.Tensor):
                            lp = lp.detach().cpu().numpy()
                        img_args["prob"] = np.exp(lp)
                        img_args["entropy"] = (-lp).mean()
                    if prm_rewards is not None:
                        img_args["prm_rewards"] = prm_rewards[i]

                    img = add_info_board(img, **img_args)
                self.frames[i].append(img)
                self.replay_images[i].append(img)

        return obs, rewards, dones, truncated, info
    
    def _save_video(self, env_idx: int, success=None):
        """Save video for a specific environment."""
        if success is None:
            success = False

        task_description = self.task_descriptions[env_idx]
        processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
        os.makedirs(self.save_dir, exist_ok=True)
        mp4_path = os.path.join(
            self.save_dir,
            f"env={env_idx}+rank={self.rank}+epi={self.total_episodes}+s={success}+"
            f"inst={processed_task_description}.mp4"
        )
        save_rollout_video(
            self.frames[env_idx],
            mp4_path=mp4_path,
        )
        self.frames[env_idx] = []
        self.replay_images[env_idx] = []
    
    def close(self):
        for i in range(self.num_envs):
            if self.frames[i]:
                self._save_video(i)
            else:
                cprint(f"[VideoWrapper] No frames to save for env {i} on close.", "yellow")
        self.env.close()

class CurriculumWrapper(gym.Wrapper):
    """
    An adaptive curriculum that selects tasks based on the agent's current capabilities, which 
    prioritizes tasks with ~50% success rate as the frontier of the agent's capabilities, 
    while maintaining exposure to both mastered and challenging tasks, improving sample 
    efficiency and generalization.
    """
    def __init__(
        self, 
        env, 
        temp: float = 1.0, 
        min_prob: float = 0.0, 
        recompute_freq: int = 10,
        enable_logging: bool = True,
        exp_dir: str = "./"
    ):
        super().__init__(env)
        self.env = env
        self.temp = temp
        self.min_prob = min_prob
        self.target_success_rate = 0.5
        self.recompute_freq = recompute_freq
        self.enable_logging = enable_logging
        self.exp_dir = exp_dir
        self.step_count = 0
        self.num_envs = env.num_envs
        self.task_descriptions = env.task_descriptions
        self.env.state_sampler = self._sample_state_with_curriculum
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info
    
    def step(self, actions, **kwargs):
        obs, rewards, dones, truncated, info = self.env.step(actions, **kwargs)
        self.step_count += 1
        if self.step_count % self.recompute_freq == 0 and self.enable_logging:
            curriculum_stats = self.get_curriculum_stats()
            if 'curriculum_stats' not in info:
                info['curriculum_stats'] = {}
            info['curriculum_stats'].update(curriculum_stats)
        
        return obs, rewards, dones, truncated, info

    def _sample_state_with_curriculum(self, task_id: int, n_states: int) -> int:
        weights = []
        state_ids = list(range(n_states))
        task_state_results = self.env.get_task_state_results()
        
        for state_id in state_ids:
            if (task_id, state_id) in task_state_results:
                results = task_state_results[(task_id, state_id)]
                success_rate = sum(results) / len(results) if results else 0.0
            else:
                success_rate = 0.0
            distance_from_target = abs(success_rate - self.target_success_rate)
            weight = 1.0 / (distance_from_target + 1e-9)
            weight = weight ** (1.0 / self.temp)
            weights.append(weight)
        total_weight = sum(weights)
        if total_weight == 0:
            return int(np.random.randint(0, n_states))
        
        probabilities = [w / total_weight for w in weights]
        # Ensure minimum sampling probability for exploration
        probabilities = [max(p, self.min_prob / n_states) for p in probabilities]
        s = sum(probabilities)
        if s <= 0:
            probabilities = [1.0 / n_states] * n_states
        else:
            probabilities = [p / s for p in probabilities]
        
        return int(np.random.choice(state_ids, p=probabilities))
    
    def get_curriculum_stats(self) -> Dict[str, Any]:
        task_state_results = self.env.get_task_state_results()
        if not task_state_results:
            return {}
        task_groups = {}
        for (task_id, state_id), results in task_state_results.items():
            if task_id not in task_groups:
                task_groups[task_id] = {}
            task_groups[task_id][state_id] = results
        
        stats = {
            'total_visited_tasks': len(task_groups),
            'total_visited_states': len(task_state_results),
        }
        for task_id, states in task_groups.items():
            state_success_rates = []
            for state_id, results in states.items():
                if results:
                    success_rate = sum(results) / len(results)
                    state_success_rates.append(success_rate)
                    # distance = abs(success_rate - self.target_success_rate)
            if state_success_rates:
                avg_success_rate = sum(state_success_rates) / len(state_success_rates)
                stats[f'task_{task_id}_avg_success_rate'] = avg_success_rate
                stats[f'task_{task_id}_visited_states'] = len(state_success_rates)
        return stats
    
    def close(self):
        self.env.close()
