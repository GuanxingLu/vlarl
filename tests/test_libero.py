"""
Usage:
    python tests/test_libero.py
"""

import os
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from PIL import Image
import numpy as np


# def test_libero_env():

def get_libero_image(obs):
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    return img

print("Starting test_libero_env...")
benchmark_dict = benchmark.get_benchmark_dict()
task_suite_name = "libero_spatial"
task_suite = benchmark_dict[task_suite_name]()
task_id = 0
task = task_suite.get_task(task_id)
task_name = task.name
task_description = task.language
task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
print(f"[info] retrieving task {task_id} from suite {task_suite_name}, the " + \
    f"language instruction is {task_description}, and the bddl file is {task_bddl_file}")

# step over the environment
env_args = {
    "bddl_file_name": task_bddl_file,
    "camera_heights": 256,
    "camera_widths": 256
}
resize_size = (224, 224)
env = OffScreenRenderEnv(**env_args)
env.seed(0)
print(f"Environment initialized with task {task_name}")
env.reset()
print("Environment reset done.")
init_states = task_suite.get_task_init_states(task_id) # for benchmarking purpose, we fix the a set of initial states
init_state_id = 0
env.set_init_state(init_states[init_state_id])

dummy_action = [0.] * 7
# for step in range(10):
for step in range(100000000):
    # action = dummy_action
    action = np.random.uniform(0, 1, size=(7,)).tolist()
    obs, reward, done, info = env.step(action)
    img = get_libero_image(obs)

    print(f"{step:02d} - action: {action}, reward: {reward}, done: {done}, info: {info}")
    
    # if step % 5 == 0:
    #     pil_img = Image.fromarray(img)
    #     img_filename = f"libero_step_{step:02d}.png"
    #     img_path = os.path.join(os.path.dirname(__file__), img_filename)
    #     pil_img.save(img_path)
    #     print(f"Saved image for step {step} to {img_path}")

env.close()
print("test_libero_env passed!")