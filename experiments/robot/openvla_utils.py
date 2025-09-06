"""Utils for evaluating the OpenVLA policy."""

import json
import os
import time
from typing import List, Optional, Tuple
import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

# Initialize important constants and pretty-printing mode in NumPy.
ACTION_DIM = 7
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

# Initialize system prompt for OpenVLA v0.1.
OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

# Flag to track if custom classes are registered
# _CUSTOM_CLASSES_REGISTERED = False
def register_custom_classes():
    # global _CUSTOM_CLASSES_REGISTERED
    # if not _CUSTOM_CLASSES_REGISTERED:
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
        # _CUSTOM_CLASSES_REGISTERED = True

def get_vla(cfg, device=DEVICE):
    """Loads and returns a VLA model from checkpoint."""
    # Load VLA checkpoint.
    print("[*] Instantiating Pretrained VLA model")
    print("[*] Loading in BF16 with Flash-Attention Enabled")

    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    register_custom_classes()

    print(f"cfg.pretrained_checkpoint: {cfg.pretrained_checkpoint}")
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.pretrained_checkpoint,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        load_in_8bit=cfg.load_in_8bit,
        load_in_4bit=cfg.load_in_4bit,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # Move model to device.
    # Note: `.to()` is not supported for 8-bit or 4-bit bitsandbytes models, but the model will
    #       already be set to the right devices and casted to the correct dtype upon loading.
    if not cfg.load_in_8bit and not cfg.load_in_4bit:
        vla = vla.to(device)

    # Load dataset stats used during finetuning (for action un-normalization).
    dataset_statistics_path = os.path.join(cfg.pretrained_checkpoint, "dataset_statistics.json")
    if os.path.isfile(dataset_statistics_path):
        with open(dataset_statistics_path, "r") as f:
            norm_stats = json.load(f)
        vla.norm_stats = norm_stats
    else:
        print(
            "WARNING: No local dataset_statistics.json file found for current checkpoint.\n"
            "You can ignore this if you are loading the base VLA (i.e. not fine-tuned) checkpoint."
            "Otherwise, you may run into errors when trying to call `predict_action()` due to an absent `unnorm_key`."
        )

    return vla


def get_processor(cfg):
    """Get VLA model's Hugging Face processor."""
    # Register OpenVLA model to HF Auto Classes to use local code
    register_custom_classes()
    
    processor = AutoProcessor.from_pretrained(
        cfg.pretrained_checkpoint, 
        # trust_remote_code=True,
        trust_remote_code=False,  # for bad network case
    )
    return processor


def crop_and_resize(image, crop_scale, batch_size):
    """
    Center-crops an image to have area `crop_scale` * (original image area), and then resizes back
    to original size. We use the same logic seen in the `dlimp` RLDS datasets wrapper to avoid
    distribution shift at test time.

    Args:
        image: TF Tensor of shape (batch_size, H, W, C) or (H, W, C) and datatype tf.float32 with
               values between [0,1].
        crop_scale: The area of the center crop with respect to the original image.
        batch_size: Batch size.
    """
    # Convert from 3D Tensor (H, W, C) to 4D Tensor (batch_size, H, W, C)
    assert image.shape.ndims == 3 or image.shape.ndims == 4
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    # Get height and width of crop
    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

    # Get bounding box representing crop
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    # Crop and then resize back up
    image = tf.image.crop_and_resize(image, bounding_boxes, tf.range(batch_size), (224, 224))

    # Convert back to 3D Tensor (H, W, C)
    if expanded_dims:
        image = image[0]

    return image

def preprocess_input_batch(obs_list: List[np.ndarray], task_label_list: List[str], 
                           pre_thought_list: Optional[List[str]] = None, center_crop: bool = False) -> Tuple[List[Image.Image], List[str]]:
    """
    Preprocess a batch of observations and task labels using the preprocess_input function.
    """
    # Ensure the length of the input lists match
    if len(obs_list) != len(task_label_list):
        raise ValueError("obs_list and task_label_list must have the same length.")
    
    if pre_thought_list is not None and len(pre_thought_list) != len(obs_list):
        raise ValueError("pre_thought_list must have the same length as obs_list if provided.")

    processed_images = []
    prompts = []

    # Process each observation and task label in the batch
    for i in range(len(obs_list)):
        obs = obs_list[i]
        task_label = task_label_list[i]
        pre_thought = pre_thought_list[i] if pre_thought_list is not None else None

        # Use the preprocess_input function for each entry
        image, prompt = preprocess_input(obs, task_label, pre_thought=pre_thought, center_crop=center_crop)

        processed_images.append(image)
        prompts.append(prompt)

    return processed_images, prompts

def preprocess_input(obs: np.ndarray, task_label: str, pre_thought: Optional[str] = None, center_crop: bool = False) -> Tuple[Image.Image, str]:
    """
    Preprocess input for VLA.
    """

    # obs["full_image"] = obs["full_image"].astype(np.uint8)
    # if isinstance(task_label, np.object_) or isinstance(task_label, np.ndarray):
    #     # HACK: convert np.object_ to str
    #     task_label = task_label[0][0].item()

    image = obs.astype(np.uint8)

    image = Image.fromarray(image)  # [H, W, C]
    image = image.convert("RGB")

    # (If trained with image augmentations) Center crop image and then resize back up to original size.
    # IMPORTANT: Let's say crop scale == 0.9. To get the new height and width (post-crop), multiply
    #            the original height and width by sqrt(0.9) -- not 0.9!
    if center_crop:
        batch_size = 1
        crop_scale = 0.9

        # Convert to TF Tensor and record original data type (should be tf.uint8)
        image = tf.convert_to_tensor(np.array(image))
        orig_dtype = image.dtype

        # Convert to data type tf.float32 and values between [0,1]
        image = tf.image.convert_image_dtype(image, tf.float32)

        # Crop and then resize back to original size
        image = crop_and_resize(image, crop_scale, batch_size)

        # Convert back to original data type
        image = tf.clip_by_value(image, 0, 1)
        image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

        # Convert back to PIL Image
        image = Image.fromarray(image.numpy())
        image = image.convert("RGB")
        # image = np.array(image) # [H, W, C]

    # Build VLA prompt
    if pre_thought is None:
        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"

    else:   # if pre_thought is given, use it as the prompt to avoid chain-of-thought reasoning from scratch
        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"
        prompt += pre_thought

    return image, prompt


def get_vla_action(vla, processor, base_vla_name, obs, task_label, unnorm_key, pre_thought=None, center_crop=False, return_thought=False):
    """Generates an action with the VLA policy."""

    # print(f"obs.shape: {obs['full_image'].shape}")

    image = Image.fromarray(obs["full_image"])
    image = image.convert("RGB")

    # (If trained with image augmentations) Center crop image and then resize back up to original size.
    # IMPORTANT: Let's say crop scale == 0.9. To get the new height and width (post-crop), multiply
    #            the original height and width by sqrt(0.9) -- not 0.9!
    if center_crop:
        batch_size = 1
        crop_scale = 0.9

        # Convert to TF Tensor and record original data type (should be tf.uint8)
        image = tf.convert_to_tensor(np.array(image))
        orig_dtype = image.dtype

        # Convert to data type tf.float32 and values between [0,1]
        image = tf.image.convert_image_dtype(image, tf.float32)

        # Crop and then resize back to original size
        image = crop_and_resize(image, crop_scale, batch_size)

        # Convert back to original data type
        image = tf.clip_by_value(image, 0, 1)
        image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

        # Convert back to PIL Image
        image = Image.fromarray(image.numpy())
        image = image.convert("RGB")

    # Build VLA prompt
    if pre_thought is None:
        if "openvla-v01" in base_vla_name:  # OpenVLA v0.1
            prompt = (
                f"{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take to {task_label.lower()}? ASSISTANT:"
            )
        else:  # OpenVLA
            prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"

    else:   # if pre_thought is given, use it as the prompt to avoid chain-of-thought reasoning from scratch
        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"
        prompt += pre_thought

    # print(f"Prompt: {prompt}")

    # Process inputs.
    inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)

    # Get action.
    if return_thought:
        action, generated_ids = vla.predict_action_cot(**inputs, unnorm_key=unnorm_key, do_sample=False)
        generated_ids = processor.batch_decode(generated_ids)[0]
    else:
        action = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
        generated_ids = None

    return action, generated_ids
