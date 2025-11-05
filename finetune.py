"""
finetune.py

Simple script for parameter-efficient fine-tuning of OpenVLA models loaded through the HuggingFace AutoClasses, using
HuggingFace PEFT library for low-rank adaptation (LoRA).

Notes & Benchmarks:
    - Requires PEFT (`pip install peft==0.11.1`)
    - LoRA fine-tuning (see parameters below -- no quantization, LoRA rank = 32, target_modules = all-linear):
        + One 48 GB GPU can fit a Batch Size of 12
        + One 80 GB GPU can fit a Batch Size of 24

Run with:
    - [Single Node Multi-GPU (= $K) ]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py
    - [Override Config Values]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py \
                                    --data_root_dir <PATH/TO/RLDS/DATASETS/DIRECTORY> \
                                    --dataset_name <DATASET_NAME> \
                                    --run_root_dir <PATH/TO/LOGS/DIR> \
                                    ...
"""

import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import draccus
import torch
import torch.distributed as dist
import tqdm
from termcolor import cprint
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers import AutoConfig, AutoImageProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

import wandb
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder, QwenPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer, ACTION_TOKENIZERS
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

# cot
from prismatic.util.cot_utils import get_cot_tags_list

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# # === Utilities ===
# # fmt: off
# def create_vision_transform(vla: nn.Module, input_size: int) -> Callable[[Image.Image], torch.Tensor]:
#     """Gets image transform for the vision encoder."""
#     data_cfg = timm.data.resolve_model_data_config(vla.vision_backbone)
#     data_cfg["input_size"] = (3, input_size, input_size)
#     return timm.data.create_transform(
#         input_size=data_cfg["input_size"],
#         interpolation=data_cfg["interpolation"],
#         mean=data_cfg["mean"],
#         std=data_cfg["std"],
#         crop_pct=1.0,           # Set to 1.0 to disable cropping
#         crop_mode="center",     # Default crop mode --> no-op when `crop_pct == 1.0`
#         is_training=False,      # Disable image_aug when loading transform; handled by RLDS dataloader
#     )
#
# # fmt: on


@dataclass
class FinetuneConfig:
    # fmt: off
    vla_path: str = "openvla/openvla-7b"  # Path to OpenVLA model (on HuggingFace Hub)

    # Directory Paths
    data_root_dir: Path = Path("datasets/open-x-embodiment")  # Path to Open-X dataset directory
    dataset_name: str = "droid_wipe"  # Name of fine-tuning dataset (e.g., `droid_wipe`)
    run_root_dir: Path = Path("runs")  # Path to directory to store logs & checkpoints
    adapter_tmp_dir: Path = Path("adapter-tmp")  # Temporary directory for LoRA weights before fusing

    # Fine-tuning Parameters
    batch_size: int = 16  # Fine-tuning batch size
    max_steps: int = 200_000  # Max number of fine-tuning steps
    save_steps: int = 5000  # Interval for checkpoint saving
    learning_rate: float = 5e-4  # Fine-tuning learning rate
    grad_accumulation_steps: int = 1  # Gradient accumulation steps
    image_aug: bool = True  # Whether to train with image augmentations
    shuffle_buffer_size: int = 100_000  # Dataloader shuffle buffer size (can reduce if OOM)
    save_latest_checkpoint_only: bool = True  # Whether to save only one checkpoint per run and
    #   continually overwrite the latest checkpoint
    #   (If False, saves all checkpoints)

    # LoRA Arguments
    use_lora: bool = True  # Whether to use LoRA fine-tuning
    lora_rank: int = 32  # Rank of LoRA weight matrix
    lora_dropout: float = 0.0  # Dropout applied to LoRA weights
    use_quantization: bool = False  # Whether to 4-bit quantize VLA for LoRA fine-tuning
    #   => CAUTION: Reduces memory but hurts performance
    merge_model: bool = False  # Whether to merge LoRA weights into model backbone
    load_model: bool = False  # Whether to load model from checkpoint to resume training

    # Tracking Parameters
    wandb_offline: bool = False  # Whether to run W&B offline
    wandb_project: str = "openvla"  # Name of W&B project to log to (use default!)
    wandb_entity: str = "stanford-voltron"  # Name of entity to log under
    run_id_note: Optional[str] = None  # Extra note for logging, Weights & Biases

    # fmt: on
    use_cot: bool = False    # Whether to fine-tune the COT model
    reasoning_dataset_path: str = "./data/modified_libero_rlds/libero_10_no_noops/1.0.0/reasonings_formatted.json"


@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")

    # [Validate] Ensure GPU Available & Set Device / Distributed Context
    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()

    # Configure Unique Experiment ID & Log Directory
    max_pet_name_len = len("openvla-7b")
    model_pet_name = cfg.vla_path.split('/')[-1][:max_pet_name_len]
    exp_id = (
        f"{model_pet_name}+{cfg.dataset_name}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"
    )
    # exp_id = (
    #     f"{model_pet_name}+{cfg.dataset_name}"
    #     f"+b32"
    #     f"+lr-{cfg.learning_rate}"
    # )   # only for debugging
    if cfg.use_lora:
        exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    if cfg.use_quantization:
        exp_id += "+q-4bit"
    if cfg.run_id_note is not None:
        exp_id += f"--{cfg.run_id_note}"
    if cfg.image_aug:
        exp_id += "--image_aug"

    # Start =>> Build Directories
    run_dir, adapter_dir = cfg.run_root_dir / exp_id, cfg.adapter_tmp_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(adapter_dir, exist_ok=True)

    # Quantization Config =>> only if LoRA fine-tuning
    quantization_config = None
    if cfg.use_quantization:
        assert cfg.use_lora, "Quantized training only supported for LoRA fine-tuning!"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )

    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Load OpenVLA Processor and Model using HF AutoClasses
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # Device Placement =>> note that BitsAndBytes automatically handles for quantized training
    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    else:
        vla = vla.to(device_id)

    # Continue training by loading the last checkpoint's weights right after creating the model
    # ref: https://github.com/openvla/openvla/issues/116
    start_gradient_step_idx = 0
    load_model = cfg.load_model and (len(os.listdir(run_dir)) >= 5) and os.path.exists(
        run_dir / "latest_checkpoint_step.txt") and os.path.exists(run_dir / "optimizer.pt")

    if load_model:
        with open(run_dir / "latest_checkpoint_step.txt", "r") as f:
            resume_iteration = int(f.read())
        start_gradient_step_idx = resume_iteration + 1
        print(f"Resuming training from iteration {resume_iteration} ...")

        # Load Model Weights
        vla = PeftModel.from_pretrained(vla, adapter_dir, is_trainable=True)
        # vla._mark_only_adapters_as_trainable()
        vla.print_trainable_parameters()
        print(f"Load weights from {run_dir} ...")

    else:
        # [LoRA] Wrap Model w/ PEFT `LoraConfig` =>> by default we set `target_modules=all-linear`
        if cfg.use_lora:
            lora_config = LoraConfig(
                r=cfg.lora_rank,
                lora_alpha=min(cfg.lora_rank, 16),
                lora_dropout=cfg.lora_dropout,
                target_modules="all-linear",
                init_lora_weights="gaussian",
            )
            vla = get_peft_model(vla, lora_config)
            vla.print_trainable_parameters()
        print(f"Training from scratch, run_dir: {run_dir}")

    # Wrap VLA in PyTorch DDP Wrapper for Multi-GPU Training
    vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)

    # Create Optimizer =>> note that we default to a simple constant learning rate!
    trainable_params = [param for param in vla.parameters() if param.requires_grad]

    if load_model:
        # Load Optimizer
        if os.path.exists(run_dir / "optimizer.pt"):
            optimizer = AdamW(trainable_params, lr=cfg.learning_rate)
            optimizer.load_state_dict(torch.load(run_dir / "optimizer.pt"))
            cprint(f"Loaded optimizer from {run_dir} ...", "green")
        else:
            cprint(f"!!!!!!!!!!!!!! optimizer file not found in {run_dir} !!!!!!!!!!!!!!", "red")
    else:
        optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    # Create Action Tokenizer
    if 'qwen' not in cfg.vla_path:
        action_tokenizer = ActionTokenizer(processor.tokenizer)
    else:
        action_tokenizer: ActionTokenizer = ACTION_TOKENIZERS["extra_action_tokenizer"](processor.tokenizer)
        cprint(f"Using extra action tokenizer for QWEN model", "yellow")

    # Load Fine-tuning Dataset =>> note that we use an RLDS-formatted dataset following Open X-Embodiment by default.
    #   =>> If you want to use a non-RLDS dataset (e.g., a standard PyTorch Dataset) see the following commented block.
    #   =>> Note that our training code does not loop over epochs because the RLDS loader does this implicitly; if using
    #       your own Dataset, make sure to add the appropriate logic to the training loop!
    #
    # ---
    # from prismatic.vla.datasets import DummyDataset
    #
    # vla_dataset = DummyDataset(
    #     action_tokenizer,
    #     processor.tokenizer,
    #     image_transform=processor.image_processor.apply_transform,
    #     prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    # )
    # ---

    # prompt_builder_fn
    if 'qwen' in cfg.vla_path:
        prompt_builder_fn = QwenPromptBuilder
        cprint(f"Using QwenPromptBuilder for QWEN model", "yellow")
    elif 'v01' in cfg.vla_path:
        prompt_builder_fn = VicunaV15ChatPromptBuilder
    else:
        prompt_builder_fn = PurePromptBuilder

    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=prompt_builder_fn,
        print_prompt_limit=0,
        use_cot=cfg.use_cot,
    )
    vla_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
        use_cot=cfg.use_cot,
        reasoning_dataset_path=cfg.reasoning_dataset_path,
    )

    # [Important] Save Dataset Statistics =>> used to de-normalize actions for inference!
    if distributed_state.is_main_process:
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)
        save_dataset_statistics(vla_dataset.dataset_statistics, adapter_dir)

    # Create Collator and DataLoader
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
    )

    # Initialize Logging =>> W&B
    if distributed_state.is_main_process:
        mode = "offline" if cfg.wandb_offline else "online"
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{exp_id}", mode=mode)

    # Deque to store recent train metrics (used for computing smoothened metrics for gradient accumulation)
    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_action_accuracies = deque(maxlen=cfg.grad_accumulation_steps)
    recent_l1_losses = deque(maxlen=cfg.grad_accumulation_steps)

    prompt_tags = get_cot_tags_list()

    # Train!
    with tqdm.tqdm(total=cfg.max_steps - start_gradient_step_idx, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(dataloader):

            # HACK: modify batch_idx, consider grad_accumulation_steps
            if start_gradient_step_idx > 0:
                batch_idx += start_gradient_step_idx * cfg.grad_accumulation_steps

            with torch.autocast("cuda", dtype=torch.bfloat16):
                output: CausalLMOutputWithPast = vla(
                    input_ids=batch["input_ids"].to(device_id),
                    attention_mask=batch["attention_mask"].to(device_id),
                    pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                    labels=batch["labels"],
                )
                loss = output.loss

            # Normalize loss to account for gradient accumulation
            normalized_loss = loss / cfg.grad_accumulation_steps

            # Backward pass
            normalized_loss.backward()

            # Compute Accuracy and L1 Loss for Logging
            # action_logits = output.logits[:, vla.module.vision_backbone.featurizer.patch_embed.num_patches: -1] # [B, 295, 32064] -> [B, 38, 32064]
            # action_preds = action_logits.argmax(dim=2)
            # action_gt = batch["labels"][:, 1:].to(action_preds.device)
            # mask = action_gt > action_tokenizer.action_token_begin_idx

            token_logits = output.logits[:, vla.module.vision_backbone.featurizer.patch_embed.num_patches : -1]
            token_preds = token_logits.argmax(dim=2)
            token_gt = batch["labels"][:, 1:].to(token_preds.device)    # shift labels to match preds
            action_mask = token_gt > action_tokenizer.action_token_begin_idx

            def get_masks(tokens, tags):
                tag_tokens = dict()

                for tag in tags:    # e.g., TASK:
                    # encoded_tags = vla.module.llm_backbone.tokenizer.encode_plus(tag, return_tensors="pt")
                    encoded_tags = processor.tokenizer(tag, return_tensors="pt")
                    tag_ids = encoded_tags["input_ids"][0]
                    tag_tokens[tag] = tag_ids[1:].to(tokens.device)

                tag_masks = dict()
                prev_tag = None
                prev_pos = 0

                def make_mask(a, b):
                    mask = torch.zeros_like(tokens)
                    mask[a:b] = 1
                    return mask

                for i in range(len(tokens) - 1):
                    for tag, tag_ids in tag_tokens.items():
                        if i + len(tag_ids) > len(tokens):
                            continue

                        if torch.all(tokens[i : i + len(tag_ids)] == tag_ids):
                            tag_masks[prev_tag] = make_mask(prev_pos, i)
                            prev_tag = tag
                            prev_pos = i + len(tag_ids)

                tag_masks[prev_tag] = make_mask(prev_pos, len(tokens))

                for tag in tags:
                    if tag not in tag_masks:
                        tag_masks[tag] = make_mask(0, 0)

                return tag_masks
            
            def get_final_masks(tokens, tags):
                final_masks = {tag: [] for tag in tags}

                for group in tokens:
                    group_masks = get_masks(group, tags)

                    for tag in tags:
                        final_masks[tag].append(group_masks[tag])

                for tag in tags:
                    final_masks[tag] = torch.stack(final_masks[tag], dim=0)

                return final_masks

            # Compute Accuracy
            correct_preds = (token_preds == token_gt) & action_mask
            action_accuracy = correct_preds.sum().float() / action_mask.sum().float()

            # Compute L1 Loss on Predicted (Continuous) Actions
            continuous_actions_pred = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(token_preds[action_mask].cpu().numpy())
            )
            continuous_actions_gt = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(token_gt[action_mask].cpu().numpy())
            )
            action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)

            # Store recent train metrics
            recent_losses.append(loss.item())
            recent_action_accuracies.append(action_accuracy.item())
            recent_l1_losses.append(action_l1_loss.item())

            # Compute gradient step index
            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps

            # Compute smoothened train metrics
            #   =>> Equal to current step metrics when not using gradient accumulation
            #   =>> Otherwise, equal to the average of metrics observed over micro-batches used for gradient accumulation
            smoothened_loss = sum(recent_losses) / len(recent_losses)
            smoothened_action_accuracy = sum(recent_action_accuracies) / len(recent_action_accuracies)
            smoothened_l1_loss = sum(recent_l1_losses) / len(recent_l1_losses)

            # Push Metrics to W&B (every 10 gradient steps)
            if distributed_state.is_main_process and gradient_step_idx % 10 == 0:

                # Dense reasoning metrics
                metrics = {}
                if cfg.use_cot:
                    final_pred_masks = get_final_masks(token_preds, prompt_tags)
                    final_gt_masks = get_final_masks(token_gt, prompt_tags)

                    # Compute accuracy for each tag
                    for tag in prompt_tags:
                        correct_tags = [0, 0]

                        for reasoning_pred, mask_pred, reasoning_gt, mask_gt in zip(
                            token_preds, final_pred_masks[tag], token_gt, final_gt_masks[tag]
                        ):
                            tag_pred = torch.masked_select(reasoning_pred, mask_pred.bool())
                            tag_gt = torch.masked_select(reasoning_gt, mask_gt.bool())

                            max_size = max(len(tag_pred), len(tag_gt))
                            tag_pred = torch.nn.functional.pad(tag_pred, (0, max_size - len(tag_pred)))
                            tag_gt = torch.nn.functional.pad(tag_gt, (0, max_size - len(tag_gt)))

                            correct_tags[0] += (tag_pred == tag_gt).sum().float()
                            correct_tags[1] += len(tag_gt)

                        if correct_tags[1] > 0:
                            tag_accuracy = correct_tags[0] / correct_tags[1]
                            # metrics.commit(**{f"{tag[:-1].lower()}_tag_accuracy": tag_accuracy})
                            metrics.update({f"{tag[:-1].lower()}_tag_accuracy": tag_accuracy})

                wandb.log(
                    {
                        "train_loss": smoothened_loss,
                        "action_accuracy": smoothened_action_accuracy,
                        "l1_loss": smoothened_l1_loss,
                        **metrics,
                    },
                    step=gradient_step_idx,
                )

            # Optimizer Step
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                progress.update()

            # Save Model Checkpoint =>> by default, only keeps the latest checkpoint, continually overwriting it!
            if gradient_step_idx > 0 and gradient_step_idx % cfg.save_steps == 0:
                if distributed_state.is_main_process:
                    print(f"Saving Model Checkpoint for Step {gradient_step_idx}")

                    # Save step indicator to indicate the latest save step
                    with open(run_dir / "latest_checkpoint_step.txt", "w") as f:
                        f.write(str(gradient_step_idx))

                    # If LoRA, we first save adapter weights, then merge into full model; otherwise, default save!
                    save_dir = adapter_dir if cfg.use_lora else run_dir

                    # if cfg.save_latest_checkpoint_only:  # default: True
                    # Save Processor & Weights
                    processor.save_pretrained(run_dir)
                    vla.module.save_pretrained(save_dir)
                    if not cfg.save_latest_checkpoint_only:
                        # Prepare to save checkpoint in new directory
                        save_grad_step_dir = Path(str(save_dir) + f"--{gradient_step_idx}_chkpt")
                        os.makedirs(save_grad_step_dir, exist_ok=True)
                        # Save dataset statistics to new directory
                        save_dataset_statistics(vla_dataset.dataset_statistics, save_grad_step_dir)
                        # Save Weights
                        vla.module.save_pretrained(save_grad_step_dir)

                    # Save optimizer
                    torch.save(optimizer.state_dict(), run_dir / "optimizer.pt")

                    print(f"Saved Model Checkpoint for Step {gradient_step_idx} at: {run_dir}")

                # Wait for processor and adapter weights to be saved by main process
                dist.barrier()

                # Merge LoRA weights into model backbone for faster inference
                #   =>> Note that merging is slow and can be done post-hoc to speed up training
                if cfg.use_lora and cfg.merge_model:
                    base_vla = AutoModelForVision2Seq.from_pretrained(
                        cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
                    )
                    merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
                    merged_vla = merged_vla.merge_and_unload()
                    if distributed_state.is_main_process:
                        # save a file to indicate the latest save step
                        with open(run_dir / "latest_checkpoint_step.txt", "w") as f:
                            f.write(str(gradient_step_idx))
                        
                        if cfg.save_latest_checkpoint_only:  # default: True
                            # Overwrite latest checkpoint
                            merged_vla.save_pretrained(run_dir)
                        else:
                            # Prepare to save checkpoint in new directory
                            checkpoint_dir = Path(str(run_dir) + f"--{gradient_step_idx}_chkpt")
                            os.makedirs(checkpoint_dir, exist_ok=True)

                            # Save dataset statistics to new directory
                            save_dataset_statistics(vla_dataset.dataset_statistics, checkpoint_dir)

                            # Save processor and model weights to new directory
                            processor.save_pretrained(checkpoint_dir)
                            merged_vla.save_pretrained(checkpoint_dir)

                        print(f"Saved Merged Model Checkpoint for Step {gradient_step_idx} at: {run_dir}")

                    # Block on Main Process Checkpointing
                    dist.barrier()

            # Stop training when max_steps is reached
            if gradient_step_idx == cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                break


if __name__ == "__main__":
    finetune()