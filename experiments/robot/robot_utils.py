"""Utils for evaluating robot policies in various environments."""

import os
import random
import time
import itertools
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import torch
import torch.nn.functional as F
from termcolor import cprint
from transformers import (
    GenerationConfig,
    PreTrainedTokenizerBase,
    LlamaTokenizerFast,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast
from transformers.image_processing_utils import BatchFeature
from transformers.tokenization_utils import PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from transformers.utils import TensorType
from accelerate.utils import is_peft_model
from experiments.robot.openvla_utils import (
    get_vla,
    get_vla_action,
)
from PIL import Image


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


def set_seed_everywhere(seed: int):
    """Sets the random seed for Python, NumPy, and PyTorch functions."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_model(cfg, device=DEVICE, wrap_diffusion_policy_for_droid=False):
    """Load model for evaluation."""
    if cfg.model_family == "openvla":
        model = get_vla(cfg, device)
    else:
        raise ValueError("Unexpected `model_family` found in config.")
    print(f"Loaded model: {type(model)}")
    return model


def get_image_resize_size(cfg):
    """
    Gets image resize size for a model class.
    If `resize_size` is an int, then the resized image will be a square.
    Else, the image will be a rectangle.
    """
    if cfg.model_family == "openvla":
        resize_size = 224
    else:
        raise ValueError("Unexpected `model_family` found in config.")
    return resize_size


def get_action(cfg, model, obs, task_label, pre_thought=None, processor=None):
    """Queries the model to get an action."""
    if cfg.model_family == "openvla":
        action, generated_ids = get_vla_action(
            model, processor, cfg.pretrained_checkpoint, obs, task_label, cfg.unnorm_key, pre_thought,
            center_crop=cfg.center_crop, return_thought=cfg.return_thought,
        )
        assert action.shape == (ACTION_DIM,)
    else:
        raise ValueError("Unexpected `model_family` found in config.")
    return action, generated_ids

# for right padding
def add_special_token(input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, pad_token_id: int = 0) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    batch_size, seq_length = input_ids.size()
    
    # Check if we need to expand tensors
    needs_expansion = False
    for i in range(batch_size):
        # if attention_mask is not None:
        #     last_valid_pos = first_true_indices(attention_mask[i]) - 1
        # elif pad_token_id is not None:
        pad_positions = (input_ids[i] == pad_token_id).nonzero(as_tuple=True)[0]
        last_valid_pos = pad_positions[0] - 1 if len(pad_positions) > 0 else seq_length - 1
        # else:
        #     last_valid_pos = seq_length - 1
            
        if last_valid_pos >= seq_length - 1:
            needs_expansion = True
            break
    
    # If any sequence needs expansion, expand all tensors
    if needs_expansion:
        new_input_ids = torch.full((batch_size, seq_length + 1), pad_token_id, dtype=input_ids.dtype, device=input_ids.device)
        new_input_ids[:, :seq_length] = input_ids
        input_ids = new_input_ids
        
        # Update sequence length after expansion
        seq_length += 1
    
    # Now add the special token
    for i in range(batch_size):
        # if attention_mask is not None:
        #     last_valid_pos = first_true_indices(attention_mask[i]) - 1
        # elif pad_token_id is not None:
        pad_positions = (input_ids[i] == pad_token_id).nonzero(as_tuple=True)[0]
        last_valid_pos = pad_positions[0] - 1 if len(pad_positions) > 0 else seq_length - 1
        # else:
        #     last_valid_pos = seq_length - 1
        
        if input_ids[i, last_valid_pos] != 29871:
            input_ids[i, last_valid_pos + 1] = 29871

    return input_ids

# def add_special_token(input_ids: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
#     ## If the special empty token ('') does not already appear after the colon (':') token in the prompt
#     ## (after "OUT:" or "ASSISTANT:"), insert it to match the inputs seen at training time
#     if not torch.all(input_ids[:, -1] == 29871):
#         special_token = torch.full((input_ids.size(0), 1), 29871, dtype=torch.long).to(input_ids.device)
#         input_ids = torch.cat((input_ids, special_token), dim=1)
#     return input_ids

# def add_special_token(input_ids: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
#     last_context_length = first_true_indices(input_ids == pad_token_id)
#     last_token_position = max(last_context_length)
#     input_ids[:, last_token_position] = 29871
#     return input_ids

def get_actions_batch(cfg, model, obs, task_label, pre_thought=None, processor=None, prompt_builder_fn=None, **kwargs):
    """
    batch generation for actions
    """
    def prompt_engineering(prompt: str) -> str:
        # Deal with different prompt builder
        prompt_builder = prompt_builder_fn("openvla")
        # Extract text from env.step output (e.g., prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:")
        lang = prompt.split("What action should the robot take to ")[1].split("?\nOut:")[0]
        conversation = [
                {"from": "human", "value": f"What action should the robot take to {lang}?"},
            ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])
        p = prompt_builder.get_prompt()
        return p
    
    prompt_list = task_label
    # image_list = [Image.fromarray(img) for img in obs]
    image_list = obs
    prompt_list = [prompt_engineering(prompt) for prompt in prompt_list]

    inputs = processor(
        prompt_list, 
        image_list, 
        padding=True,
        # padding_side='left'  # Add left padding for llama batch inference bug, ref: https://github.com/huggingface/transformers/issues/25420#issuecomment-1775317535
        padding_side='right'
    ).to(DEVICE, dtype=torch.bfloat16)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    pixel_values = inputs["pixel_values"]

    input_ids = add_special_token(input_ids, attention_mask, pad_token_id=processor.tokenizer.pad_token_id)  # 
    attention_mask = input_ids != processor.tokenizer.pad_token_id

    print(f"input_ids.shape: {input_ids.shape}")
    print(f"input_ids: {input_ids}")
    print(f"attention_mask.shape: {attention_mask.shape}")
    print(f"pixel_values.shape: {pixel_values.shape}")
    print(f"processor.tokenizer.pad_token_id: {processor.tokenizer.pad_token_id}")

    # support batch inference
    generated_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        # max_new_tokens=1024, 
        max_new_tokens=8,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True, # the output logits match well with generated_ids.sequences
        **kwargs
    )
    # [B, input_l+response_l], bos token: 1, eos token: 2, pad token: 32000

    context_length = input_ids.shape[1]
    response_token_ids = generated_ids.sequences[:, context_length:]    # [B, max_new_tokens], with possible padding
    action_dim = model.get_action_dim(cfg.unnorm_key)

    print(f"response_token_ids: {response_token_ids}")

    stop_token_id = processor.tokenizer.eos_token_id
    trunc_idxs = first_true_indices(response_token_ids == stop_token_id).unsqueeze(-1)

    # print(f"response_token_ids.shape: {response_token_ids}")

    predicted_action_token_ids = torch.zeros((response_token_ids.shape[0], action_dim), dtype=torch.long).to(DEVICE)
    action_log_prob = None
    for i in range(response_token_ids.shape[0]):
        predicted_action_token_ids[i] = response_token_ids[i, trunc_idxs[i] - action_dim:trunc_idxs[i]] # [B, action_dim], without padding
        # other outputs
        # pi_logits = generated_ids.scores[:][i, context_length:trunc_idxs[i]]    # [response_l, vocab_size]
        # pi_log_softmax = torch.log_softmax(pi_logits, dim=-1)
        # token_log_probs = torch.gather(pi_log_softmax, -1, action_tokens.unsqueeze(-1)).squeeze(-1)
        # action_log_prob = token_log_probs.sum()
        
    # numpy
    predicted_action_token_ids_np = predicted_action_token_ids.cpu().numpy()    
    discretized_actions = model.vocab_size - predicted_action_token_ids_np
    discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=model.bin_centers.shape[0] - 1)
    normalized_actions = model.bin_centers[discretized_actions]

    # Unnormalize actions
    action_norm_stats = model.get_action_stats(cfg.unnorm_key)
    mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))    # [7,]
    action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
    robot_action = np.where(
        mask,
        0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
        normalized_actions,
    )   # [len(valid_ids), 7]

    return robot_action, predicted_action_token_ids, action_log_prob

def get_actions(cfg, model, obs, task_label, pre_thought=None, processor=None, prompt_builder_fn=None, **kwargs):
    """
    batch generation for actions
    """
    def prompt_engineering(prompt: str) -> str:
        # Deal with different prompt builder
        prompt_builder = prompt_builder_fn("openvla")
        # Extract text from env.step output (e.g., prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:")
        lang = prompt.split("What action should the robot take to ")[1].split("?\nOut:")[0]
        conversation = [
                {"from": "human", "value": f"What action should the robot take to {lang}?"},
            ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])
        p = prompt_builder.get_prompt()
        return p
    
    prompt_list = task_label
    # image_list = [Image.fromarray(img) for img in obs]
    image_list = obs
    prompt_list = [prompt_engineering(prompt) for prompt in prompt_list]
    batch_size = len(image_list)

    action_dim = model.get_action_dim(cfg.unnorm_key)
    stop_token_id = processor.tokenizer.eos_token_id if not isinstance(processor.tokenizer, Qwen2TokenizerFast) else 151645

    predicted_action_token_ids = torch.zeros((len(image_list), action_dim), dtype=torch.long).to(DEVICE)
    action_tokens = None
    action_log_prob = None

    for i in range(batch_size):
        inputs = processor(prompt_list[i], image_list[i]).to(DEVICE, dtype=torch.bfloat16)
        input_ids = inputs["input_ids"]
        pixel_values = inputs["pixel_values"]

        # critical, some tokenizers have different numbers of "end tokens".
        num_end_tokens = 1
        if isinstance(processor.tokenizer, LlamaTokenizerFast):
            # If the special empty token ('') does not already appear after the colon (':') token in the prompt
            # (after "OUT:" or "ASSISTANT:"), insert it to match the inputs seen at training time
            if not torch.all(input_ids[:, -1] == 29871):
                input_ids = torch.cat(
                    (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
                )
        elif isinstance(processor.tokenizer, Qwen2TokenizerFast):
            # do nothing here. I think...
            # Qwen has <|im_end|><|endoftext|> for example
            num_end_tokens = 2
            pass
        else:
            raise ValueError(f"Unsupported `tokenizer` type = {type(processor.tokenizer)}")

        generated_ids = model.generate(
            input_ids=input_ids,
            # attention_mask=attention_mask[i:i+1],
            pixel_values=pixel_values,
            max_new_tokens=1024, 
            # do_sample=False,
            do_sample=True,
            return_dict_in_generate=True,
            output_scores=True, # the output logits match well with generated_ids.sequences
            **kwargs
        )
        # [B=1, input_l+response_l], bos token: 1, eos token: 2, pad token: 32000
        context_length = input_ids.shape[1]
        response_token_ids = generated_ids.sequences[0, context_length:]    # [B=1, max_new_tokens], with possible padding
        trunc_idxs = first_true_indices(response_token_ids == stop_token_id)

        # predicted_action_token_ids[i] = response_token_ids[trunc_idxs + 1 - action_dim - num_end_tokens:trunc_idxs + 1 - num_end_tokens] # [B, action_dim], without padding and eos token
        try:
            predicted_action_token_ids[i] = response_token_ids[trunc_idxs - action_dim:trunc_idxs]
        except:
            cprint(f"Wrong response_token_ids: {response_token_ids}", "red")
        # other outputs
        # pi_logits = generated_ids.scores[:][i, context_length:trunc_idxs[i]]    # [response_l, vocab_size]
        # pi_log_softmax = torch.log_softmax(pi_logits, dim=-1)
        # token_log_probs = torch.gather(pi_log_softmax, -1, action_tokens.unsqueeze(-1)).squeeze(-1)
        # action_log_prob = token_log_probs.sum()
        
    # numpy
    vocab_size = model.vocab_size if not isinstance(processor.tokenizer, Qwen2TokenizerFast) else len(processor.tokenizer)
    predicted_action_token_ids_np = predicted_action_token_ids.cpu().numpy()    
    discretized_actions = vocab_size - predicted_action_token_ids_np
    discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=model.bin_centers.shape[0] - 1)
    normalized_actions = model.bin_centers[discretized_actions]

    # Unnormalize actions
    action_norm_stats = model.get_action_stats(cfg.unnorm_key)
    mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))    # [7,]
    action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
    robot_action = np.where(
        mask,
        0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
        normalized_actions,
    )   # [len(valid_ids), 7]

    return robot_action, predicted_action_token_ids, action_log_prob

def normalize_gripper_action(action, binarize=True):
    """
    Changes gripper action (last dimension of action vector) from [0,1] to [-1,+1].
    Necessary for some environments (not Bridge) because the dataset wrapper standardizes gripper actions to [0,1].
    Note that unlike the other action dimensions, the gripper action is not normalized to [-1,+1] by default by
    the dataset wrapper.

    Normalization formula: y = 2 * (x - orig_low) / (orig_high - orig_low) - 1
    """
    # Create a copy of the input array to make it writable
    action = np.array(action, copy=True)
    
    # Just normalize the last action to [-1,+1].
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 2 * (action[..., -1] - orig_low) / (orig_high - orig_low) - 1

    if binarize:
        # Binarize to -1 or +1.
        action[..., -1] = np.sign(action[..., -1])

    return action

def invert_gripper_action(action):
    """
    Flips the sign of the gripper action (last dimension of action vector).
    This is necessary for some environments where -1 = open, +1 = close, since
    the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.
    """
    action[..., -1] = action[..., -1] * -1.0
    return action

# ----------------------------------------------------------------------------
def disable_dropout_in_model(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            # cprint(f"Disabling dropout in module: {module}", "yellow")
            module.p = 0

def first_true_indices(bools: torch.Tensor, dtype=torch.long) -> torch.Tensor:
    """
    Finds the index of the first `True` value in each row of a boolean tensor. If no `True` value exists in a row,
    it returns the length of the row.
    """
    row_len = bools.size(-1)
    zero_or_index = row_len * (~bools).type(dtype) + torch.arange(row_len, dtype=dtype, device=bools.device)
    return torch.min(zero_or_index, dim=-1).values

def truncate_response(stop_token_id: int, pad_token_id: int, responses: torch.Tensor):
    """
    Truncates the responses at the first occurrence of the stop token, filling the rest with pad tokens.
    Args:
        stop_token_id (`int`):
            The token ID representing the stop token where truncation occurs.
        pad_token_id (`int`):
            The token ID representing the pad token used to fill the truncated responses.
        responses (`torch.Tensor`):
            The tensor containing the responses to be truncated.
    Returns:
        `torch.Tensor`:
            The truncated responses tensor with pad tokens filled after the stop token.
    """
    trunc_idxs = first_true_indices(responses == stop_token_id).unsqueeze(-1)
    new_size = [1] * (len(responses.size()) - 1) + [responses.shape[1]]
    idxs = torch.arange(responses.shape[1], device=responses.device).view(*new_size)
    postprocessed_responses = torch.masked_fill(responses, idxs > trunc_idxs, pad_token_id)
    return postprocessed_responses

def exact_div(a, b, custom_error_message=""):
    q = a // b
    if a != q * b:
        raise ValueError(f"{custom_error_message}, inexact division: {a} / {b} = {a / b}")
    return q

@contextmanager
def unwrap_model_for_generation(
    model, accelerator, is_peft_model: bool = False
):
    """Context manager to unwrap a model for generation.
    For ZeRO-3 models, we gather the weights once to speed up generation.
    """
    unwrapped_model = accelerator.unwrap_model(model)
    # if is_peft_model:
    #     unwrapped_model.pretrained_model.disable_adapter()

    yield unwrapped_model

def pad(tensors: list[torch.Tensor], padding_value: int = 0, padding_side: str = "right") -> torch.Tensor:
    # Determine the maximum shape for each dimension
    output_shape = np.max([t.shape for t in tensors], 0).tolist()

    # Create an output tensor filled with the padding value
    output = torch.full((len(tensors), *output_shape), padding_value, dtype=tensors[0].dtype, device=tensors[0].device)

    for i, t in enumerate(tensors):
        # Determine the slice for the sequence dimension
        if padding_side == "left":
            seq_slice = slice(output_shape[0] - t.shape[0], output_shape[0])
        elif padding_side == "right":
            seq_slice = slice(0, t.shape[0])
        else:
            raise ValueError("padding_side must be 'left' or 'right'")

        slices = (seq_slice,) + tuple(slice(0, s) for s in t.shape[1:])
        output[i][slices] = t

    return output

def generate(
    lm_backbone: torch.nn.Module, queries: torch.Tensor, pad_token_id: int, generation_config: GenerationConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates sequences from the language model backbone in a way that does not affect padding tokens.

    Args:
        lm_backbone (`torch.nn.Module`):
            The language model backbone used for generation.
        queries (`torch.Tensor`):
            The tensor containing the input queries.
        pad_token_id (`int`):
            The token ID representing the pad token.
        generation_config (`GenerationConfig`):
            The configuration for the generation process.

    Returns:
        tuple:
            - `generated_sequences` (`torch.Tensor`):
                The concatenated tensor of input queries and generated sequences.
            - `logits` (`torch.Tensor`):
                The logits output from the generation process.
    """
    context_length = queries.shape[1]
    attention_mask = queries != pad_token_id
    input_ids = torch.masked_fill(queries, ~attention_mask, 0)
    output = lm_backbone.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        # position_ids=attention_mask.cumsum(1) - attention_mask.long(), # not needed: already adjusted in generations
        # https://github.com/huggingface/transformers/blob/ac33aeeeee2a7a89b89c93c2962e6feb90daef0a/src/transformers/models/gpt2/modeling_gpt2.py#L1227-L1250
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
    )
    logits = torch.stack(output.scores, 1)
    return torch.cat((queries, output.sequences[:, context_length:]), dim=1), logits

@torch.no_grad()
def batch_generation(
    model: torch.nn.Module,
    queries: torch.Tensor,
    local_rollout_forward_batch_size: int,
    pad_token_id: int,
    generation_config,
):
    query_responses = []
    logitss = []
    batch_size = queries.shape[0]
    for i in range(0, batch_size, local_rollout_forward_batch_size):
        query = queries[i : i + local_rollout_forward_batch_size]
        query_response, logits = generate(
            model,
            query,
            pad_token_id,
            generation_config,
        )
        query_responses.append(query_response)
        logitss.append(logits)

    # padding tensors
    padded_query_responses = pad(query_responses, padding_value=pad_token_id, padding_side="right")
    padded_logitss = pad(logitss, padding_value=0, padding_side="right")

    # reshaping
    padded_query_responses = padded_query_responses.view(-1, padded_query_responses.shape[-1])[:batch_size]
    padded_logitss = padded_logitss.view(-1, *padded_logitss.shape[2:])[:batch_size]

    return padded_query_responses, padded_logitss


def process_with_padding_side(
        processor,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]],
        images: Union[Image.Image, List[Image.Image]],
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Optional[Union[bool, str, TruncationStrategy]] = None,
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
        padding_side: Optional[str] = None,
    ) -> BatchFeature:
    """
    Preprocess a given (batch) of text/images for a Prismatic VLM; forwards text to the underlying LLM's tokenizer,
    forwards images to PrismaticImageProcessor.

    @param text: The (batch) of text to encode; must be a string or list of strings.
    @param images: A (batch of) PIL.Image.Image instance(s) to preprocess.
    @param padding: Sequence padding strategy (if multiple specified) in < True = "longest" | "max_length" | False >
    @param truncation: Truncation strategy for the output sequences; requires `max_length` to be specified
    @param max_length: Maximum length (in tokens) to truncate
    @param return_tensors: Type of return tensors (usually "pt" or TensorType.PYTORCH)

    @return: BatchFeature with keys for `input_ids`, `attention_mask` and `pixel_values`.
    """
    pixel_values = processor.image_processor(images, return_tensors="pt")["pixel_values"]
    text_inputs = processor.tokenizer(
        text, return_tensors=return_tensors, padding=padding, truncation=truncation, max_length=max_length, padding_side=padding_side
    )
    # [Validate] Need same number of images and text inputs!
    if pixel_values.shape[0] != text_inputs.input_ids.shape[0]:
        raise ValueError("Batch is malformed; expected same number of images and text inputs!")
    return BatchFeature(data={**text_inputs, "pixel_values": pixel_values})

def forward(
        model: torch.nn.Module,
        query_response: torch.LongTensor | Tuple[torch.LongTensor, torch.FloatTensor],
        pixel_values: torch.FloatTensor,
        response: torch.LongTensor,
        pad_token_id: int,
        context_length: List[int],
        temperature: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        model (`torch.nn.Module`):
            The model used to compute the log probability.
        query_response (`torch.LongTensor`):
            The tensor containing the query responses. [B, prompt_length + response_length]
        pixel_values (`torch.FloatTensor`):
            The tensor containing the pixel values. [B, C, H, W]
        response (`torch.LongTensor`):
            The tensor containing the response. [B, response_length]
        pad_token_id (`int`):
            The token ID representing the pad token.
        context_length (`List[int]`):
            The length of the context in the query responses. [B]
        temperature (`float`):
            The temperature for the log probability.

    Returns:
        logprob (`torch.Tensor`):
            The log probability of the query-response pairs. [B, max_response_length]
        all_logits (`torch.Tensor`):
            The logits of the query-response pairs. [B, max_response_length, vocab_size]
    """
    attention_mask = query_response != pad_token_id
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output: CausalLMOutputWithPast = model(
            input_ids=query_response,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            return_dict=True,
        )
    # https://github.com/OpenRLHF/OpenRLHF/pull/634
    output["logits"] = output["logits"].to(torch.float32)

    # Get the maximum response length to handle variable length responses
    max_response_len = response.size(1)
    batch_size = query_response.shape[0]
    vocab_size = output.logits.size(-1)
    
    # Create a tensor to hold all logits with proper padding
    all_logits = torch.zeros(batch_size, max_response_len, vocab_size, 
                            device=output.logits.device, dtype=output.logits.dtype)
    
    # Extract logits for each sample based on its context length
    for i in range(batch_size):
        sample_logits = output.logits[i, context_length[i] - 1 : context_length[i] - 1 + response.size(1)]
        all_logits[i, :sample_logits.size(0)] = sample_logits

    all_logits.div_(temperature + 1e-7)
    
    all_logprob = F.log_softmax(all_logits, dim=-1)
    logprob = torch.gather(all_logprob, 2, response.unsqueeze(-1)).squeeze(-1)

    return (
        logprob,    # [B, max_response_length]
        all_logits, # [B, max_response_length, vocab_size]
    )

def get_reward(
    model: torch.nn.Module, queries: torch.Tensor, pixel_values: torch.Tensor, pad_token_id: int, context_length: int=None
) -> torch.Tensor:
    """
    Computes the reward logits and the rewards for a given model and query responses.

    Args:
        model (`torch.nn.Module`):
            The model used to compute the reward logits.
        query_responses (`torch.Tensor`):
            The tensor containing the query responses.
        pixel_values (`torch.Tensor`):
            The tensor containing the pixel values.
        pad_token_id (`int`):
            The token ID representing the pad token.
        context_length (`int`):
            The length of the context in the query responses.

    Returns:
        tuple:
            - `reward_logits` (`torch.Tensor`):
                The logits for the reward model.
            - `final_rewards` (`torch.Tensor`):
                The final rewards for each query response.
            - `sequence_lengths` (`torch.Tensor`):
                The lengths of the sequences in the query responses.
    """
    attention_mask = queries != pad_token_id
    # lm_backbone = getattr(model, model.base_model_prefix)
    # input_ids = torch.masked_fill(queries, ~attention_mask, 0)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        # if is_peft_model(model):
        #     print("[Critic] is peft model, disable adapter")
        #     with model.disable_adapter():
        #         reward_logits = model(
        #             input_ids=queries,
        #             pixel_values=pixel_values,
        #             attention_mask=attention_mask,
        #         )
        # else:
        reward_logits = model(
            input_ids=queries,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
        )
    return reward_logits

# qwen2.5vl
# def get_reward(
#     model: torch.nn.Module, text_list: List[str], image_list: List[Image.Image], pad_token_id: int, context_length: int
# ) -> torch.Tensor:
#     """
#     Computes the reward logits and the rewards for a given model and query responses.

#     Args:
#         model (`torch.nn.Module`):
#             The model used to compute the reward logits.
#         query_responses (`torch.Tensor`):
#             The tensor containing the query responses.
#         pixel_values (`torch.Tensor`):
#             The tensor containing the pixel values.
#         pad_token_id (`int`):
#             The token ID representing the pad token.
#         context_length (`int`):
#             The length of the context in the query responses.

#     Returns:
#         tuple:
#             - `reward_logits` (`torch.Tensor`):
#                 The logits for the reward model.
#             - `final_rewards` (`torch.Tensor`):
#                 The final rewards for each query response.
#             - `sequence_lengths` (`torch.Tensor`):
#                 The lengths of the sequences in the query responses.
#     """
#     reward_logits = model(
#         text_list=text_list,
#         image_list=image_list,
#     )
#     return reward_logits

def masked_mean(values: torch.Tensor, mask: torch.Tensor, axis: Optional[bool] = None) -> torch.Tensor:
    """Compute mean of tensor with a masked values."""
    if axis is not None:
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis)
    else:
        return (values * mask).sum() / mask.sum()


def masked_var(values: torch.Tensor, mask: torch.Tensor, unbiased: bool = True) -> torch.Tensor:
    """Compute variance of tensor with masked values."""
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values**2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            raise ValueError(
                "The sum of the mask is zero, which can happen when `mini_batch_size=1`;"
                "try increase the `mini_batch_size` or `gradient_accumulation_steps`"
            )
        # note that if mask_sum == 1, then there is a division by zero issue
        # to avoid it you just need to use a larger minibatch_size
        bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction
    return variance


def masked_whiten(values: torch.Tensor, mask: torch.Tensor, shift_mean: bool = True) -> torch.Tensor:
    """Whiten values with masked values."""
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


def remove_padding(sequences, pad_token_id):
    return [[inneritem for inneritem in item if inneritem != pad_token_id] for item in sequences]

# ----------------------------------------------------------------------------
# Quality of life utilities
def format_value(value):
    if isinstance(value, float):
        if abs(value) < 1e-2:
            return f"{value:.2e}"
        return f"{value:.2f}"
    return str(value)

def print_rich_single_line_metrics(metrics):
    # Create main table
    table = Table(show_header=False, box=None)
    table.add_column("Category", style="cyan")
    table.add_column("Values", style="magenta")

    # Group metrics by their prefix
    grouped_metrics = defaultdict(list)
    for key, value in metrics.items():
        category = key.split("/")[0] if "/" in key else "other"
        grouped_metrics[category].append((key, value))

    # Sort groups by category name
    for category in sorted(grouped_metrics.keys()):
        values = grouped_metrics[category]
        value_strings = []
        for key, value in values:
            # Use the last part of the key as the display name
            display_name = key.split("/")[-1]
            value_strings.append(f"{display_name}: {format_value(value)}")

        # Join all values for this category into a single string
        values_str = " | ".join(value_strings)
        table.add_row(category, values_str)

    # Create a panel with the table
    panel = Panel(
        table,
        title="Metrics",
        expand=False,
        border_style="bold green",
    )

    # Print the panel
    rprint(panel)


if __name__ == "__main__":
    print(f"{format_value(1e-4)=}")
    print(f"{format_value(1e-5)=}")
    print(f"{format_value(1e-6)=}")
