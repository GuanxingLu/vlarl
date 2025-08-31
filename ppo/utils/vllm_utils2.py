# Copied and Adapted from
# https://github.com/OpenRLHF/OpenRLHF and https://github.com/allenai/open-instruct
# which has the following license:
# Copyright 2023 OpenRLHF
# Copyright 2024 AllenAI
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
import time
from datetime import timedelta
from typing import Any, Optional, Union, Dict

import ray
import numpy as np
import torch
import torch.distributed
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from torch.distributed.distributed_c10d import (
    Backend,
    PrefixStore,
    Store,
    _new_process_group_helper,
    _world,
    default_pg_timeout,
    rendezvous,
)
from vllm import LLM
from vllm.worker.worker import Worker

@ray.remote
def get_all_env_variables():
    import os

    return os.environ

# Copy from pytorch to allow creating multiple main groups.
# https://github.com/pytorch/pytorch/blob/main/torch/distributed/distributed_c10d.py
def init_process_group(
    backend: Union[str, Backend] = None,
    init_method: Optional[str] = None,
    timeout: Optional[timedelta] = None,
    world_size: int = -1,
    rank: int = -1,
    store: Optional[Store] = None,
    group_name: str = None,
    pg_options: Optional[Any] = None,
):
    assert (store is None) or (init_method is None), "Cannot specify both init_method and store."

    if store is not None:
        assert world_size > 0, "world_size must be positive if using store"
        assert rank >= 0, "rank must be non-negative if using store"
    elif init_method is None:
        init_method = "env://"

    if backend:
        backend = Backend(backend)
    else:
        backend = Backend("undefined")

    if timeout is None:
        timeout = default_pg_timeout

    # backward compatible API
    if store is None:
        rendezvous_iterator = rendezvous(init_method, rank, world_size, timeout=timeout)
        store, rank, world_size = next(rendezvous_iterator)
        store.set_timeout(timeout)

        # Use a PrefixStore to avoid accidental overrides of keys used by
        # different systems (e.g. RPC) in case the store is multi-tenant.
        store = PrefixStore(group_name, store)

    # NOTE: The pg_options parameter was renamed into backend_options in PyTorch 2.6.0
    # https://github.com/pytorch/pytorch/commit/a0c7029a75628cd5fa8df83c0de0ea98ee7fd844
    # We need to determine the appropriate parameter name based on PyTorch version
    pg_options_param_name = "backend_options" if str(torch.__version__) >= "2.6" else "pg_options"
    pg, _ = _new_process_group_helper(
        world_size,
        rank,
        [],
        backend,
        store,
        group_name=group_name,
        **{pg_options_param_name: pg_options},
        timeout=timeout,
    )

    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}

    return pg


class WorkerWrap(Worker):
    def init_process_group(
        self, master_address, master_port, rank_offset, world_size, group_name, backend="nccl", use_ray=False
    ):
        """Init torch process group for model weights update"""
        assert torch.distributed.is_initialized(), f"default torch process group must be initialized"
        assert group_name != "", f"group name must not be empty"

        rank = torch.distributed.get_rank() + rank_offset
        if use_ray:
            import ray.util.collective as collective

            collective.init_collective_group(world_size=world_size, rank=rank, backend=backend, group_name=group_name)
            self._model_update_group = group_name
        else:
            self._model_update_group = init_process_group(
                backend=backend,
                init_method=f"tcp://{master_address}:{master_port}",
                world_size=world_size,
                rank=rank,
                group_name=group_name,
            )
        self._model_update_with_ray = use_ray
        print(
            f"init_process_group: master_address={master_address}, master_port={master_port}, ",
            f"rank={rank}, world_size={world_size}, group_name={group_name}",
        )

    def update_weight(self, name, dtype, shape, empty_cache=False):
        """Broadcast weight to all vllm workers from source rank 0 (actor model)"""
        # if torch.distributed.get_rank() == 0:
        #     print(f"update weight: {name}, dtype: {dtype}, shape: {shape}")

        assert dtype == self.model_config.dtype, f"mismatch dtype: src {dtype}, dst {self.model_config.dtype}"
        weight = torch.empty(shape, dtype=dtype, device="cuda")
        if self._model_update_with_ray:
            import ray.util.collective as collective

            collective.broadcast(weight, 0, group_name=self._model_update_group)
        else:
            torch.distributed.broadcast(weight, 0, group=self._model_update_group)

        self.model_runner.model.load_weights(weights=[(name, weight)])

        del weight
        # TODO: should we empty cache if all weights have updated?
        # if empty_cache:
        #     torch.cuda.empty_cache()


# @ray.remote
class LLMRayActor:
    def __init__(self, *args, **kwargs):
        if kwargs.get("distributed_executor_backend") == "ray":
            # a hack to make the script work.
            # stop ray from manipulating CUDA_VISIBLE_DEVICES
            # at the top-level when the distributed_executor_backend is ray.
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)

        self.llm = LLM(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.llm.generate(*args, **kwargs)

    def init_process_group(self, master_address, master_port, rank_offset, world_size, group_name, backend, use_ray):
        return self.llm.collective_rpc(
            "init_process_group",
            args=(master_address, master_port, rank_offset, world_size, group_name, backend, use_ray),
        )

    def update_weight(self, name, dtype, shape, empty_cache=False):
        return self.llm.collective_rpc("update_weight", args=(name, dtype, shape, empty_cache))
    


@ray.remote
class VLARayActor(LLMRayActor):
    def __init__(self, *args, **kwargs):
        # norm_stats = kwargs.pop("norm_stats") if "norm_stats" in kwargs else self.config.norm_stats
        super().__init__(*args, **kwargs)
        # self.config = self.llm.vllm_config.model_config.hf_config
        self.config = self.llm.llm_engine.get_model_config().hf_config

        self.eos_token_id = self.llm.llm_engine.input_preprocessor.get_eos_token_id()
        self.norm_stats = self.config.norm_stats
        
        # Compute action bins
        self.bins = np.linspace(-1, 1, self.config.n_action_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        # Compute vocab size for de-tokenization -- revert added "multiple of"
        self.vocab_size = self.config.text_config.vocab_size - self.config.pad_to_multiple_of

    def predict_action(self, *args, **kwargs):
        unnorm_key = kwargs.pop("unnorm_key")
        action_dim = self.get_action_dim(unnorm_key)

        outputs = self.generate(*args, **kwargs)
        response_ids = []
        response_logprobs = []
        for output in outputs:
            for out in output.outputs:
                response_ids.append(list(out.token_ids))
                response_logprobs.append([out.logprobs[id][token].logprob for id, token in enumerate(out.token_ids)])

        # Pad sequences to same length with stop token
        max_len = max(len(ids) for ids in response_ids)
        stop_token_id = self.eos_token_id
        response_token_ids = np.array([
            ids + [stop_token_id] * (max_len - len(ids)) 
            for ids in response_ids
        ])
        # Find first occurrence of stop token
        trunc_idxs = np.argmax(response_token_ids == stop_token_id, axis=1)
        predicted_action_token_ids = np.zeros((response_token_ids.shape[0], action_dim), dtype=np.int64)
        for i in range(response_token_ids.shape[0]):
            if trunc_idxs[i] >= action_dim:
                predicted_action_token_ids[i] = response_token_ids[i, trunc_idxs[i] - action_dim:trunc_idxs[i]]

        discretized_actions = self.vocab_size - predicted_action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
        normalized_actions = self.bin_centers[discretized_actions]
        # Unnormalize actions
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        return actions, response_ids, response_logprobs

    @staticmethod
    def _check_unnorm_key(norm_stats: Dict[str, Dict[str, Any]], unnorm_key: Optional[str]) -> str:
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key

    def get_action_dim(self, unnorm_key: Optional[str] = None) -> int:
        """Get the dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return len(self.norm_stats[unnorm_key]["action"]["q01"])

    def get_action_stats(self, unnorm_key: Optional[str] = None) -> Dict[str, Any]:
        """Get all the logged statistics for the given dataset."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]["action"]



def create_vllm_engines(
    num_engines: int,
    tensor_parallel_size: int,
    enforce_eager: bool,
    pretrain: str,  # i.e., model name or path
    revision: str,
    seed: int,
    enable_prefix_caching: bool,
    max_model_len: int,
    **kwargs,
):
    import vllm
    
    assert vllm.__version__ >= "0.7.0", "Current version of vLLM is not supported. Please use vLLM >= 0.7.0."

    vllm_engines = []
    # When tensor_parallel_size=1, vLLM init model in LLMEngine directly, assign 1 GPU for it.
    num_gpus = int(tensor_parallel_size == 1)
    distributed_executor_backend = "uni" if tensor_parallel_size == 1 else "ray"
    for i in range(num_engines):
        scheduling_strategy = None
        if tensor_parallel_size > 1:
            bundles = [{"GPU": 1, "CPU": 1}] * tensor_parallel_size
            pg = placement_group(bundles)
            ray.get(pg.ready())

            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=pg, placement_group_capture_child_tasks=True, placement_group_bundle_index=0
            )

        print(f"vllm: {num_gpus=}, {num_engines=}")

        vllm_engines.append(
            VLARayActor.options(
                num_cpus=0,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
            ).remote(
                pretrain,
                worker_cls="ppo.utils.vllm_utils2.WorkerWrap",
                trust_remote_code=True,
                # trust_remote_code=False,
                tensor_parallel_size=tensor_parallel_size,
                enforce_eager=enforce_eager,
                dtype="bfloat16",
                seed=seed + i,
                enable_prefix_caching=enable_prefix_caching,
                max_model_len=max_model_len,
                distributed_executor_backend=distributed_executor_backend,
                **kwargs,
            )
        )

    return vllm_engines



if __name__ == "__main__":
    vla_actor = VLARayActor.remote(
        "./MODEL/openvla-7b", 
        tensor_parallel_size=1,
        trust_remote_code=True,
    )
    print("[INFO] VLARayActor successfully created.")

    # Create sampling parameters
    from vllm import SamplingParams
    generation_config = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=64,
        include_stop_str_in_output=True,
        n=1,
    )

    # Load test data
    from PIL import Image
    data_path = "debug"
    
    with open(f"{data_path}/lang.txt", "r") as f:
        lang = f.read()
    image = Image.open(f"{data_path}/image.png")

    # Prepare input with multimodal data
    test_input = [{
        "prompt": "<PAD>" + lang.split("Out: ")[0] + "Out: ",
        "multi_modal_data": {"image": image},
    }]

    # Test action prediction
    actions, response_ids, response_logprobs = ray.get(vla_actor.predict_action.remote(
        test_input,
        sampling_params=generation_config,
        unnorm_key="austin_buds_dataset_converted_externally_to_rlds"  # Should match your normalization stats key
    ))
    
    print(f"Predicted actions: {actions}")
    print(f"Predicted token_ids: {response_ids}")
    print(f"Predicted logprobs: {response_logprobs}")
