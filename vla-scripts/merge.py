"""
merge.py
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import draccus
import torch
from peft import PeftModel
from transformers import AutoModelForVision2Seq, AutoProcessor
from transformers import AutoConfig, AutoImageProcessor
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor


@dataclass
class MergeConfig:
    # fmt: off
    vla_path: str = "openvla/openvla-7b"  # Path to OpenVLA model (on HuggingFace Hub)

    # Directory Paths
    run_root_dir: Path = Path("runs")  # Path to directory to store logs & checkpoints
    adapter_tmp_dir: Path = Path("adapter-tmp")  # Temporary directory for LoRA weights before fusing

    # HACK
    copy_needed_files: bool = True
    # fmt: on

@draccus.wrap()
def merge(cfg: MergeConfig) -> None:
    # Start =>> Build Directories
    # run_dir, adapter_dir = cfg.run_root_dir / exp_id, cfg.adapter_tmp_dir / exp_id
    run_dir, adapter_dir = cfg.run_root_dir, cfg.adapter_tmp_dir

    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    base_vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    )
    merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
    merged_vla = merged_vla.merge_and_unload()

    # Overwrite latest checkpoint
    merged_vla.save_pretrained(run_dir)

    del merged_vla, base_vla

    # move the needed files to the run directory
    if cfg.copy_needed_files:
        needed_files = [
            # ".gitattributes",
            "added_tokens.json",
            # "config.json",
            # "configuration_prismatic.py",
            # "dataset_statistics.json",
            # "generation_config.json",
            # "model.safetensors.index.json",
            # "modeling_prismatic.py",
            "preprocessor_config.json",
            # "processing_prismatic.py",
            "README.md",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "tokenizer.model",
        ]
        # copy needed files into the run dir
        for file in needed_files:
            file_path = os.path.join(adapter_dir, file)
            os.system(f"cp {file_path} {run_dir}")
            print(f"Copyed {file_path} to {run_dir}")

    print(f"Merged Model Checkpoint at {run_dir}")

    
if __name__ == "__main__":
    merge()
