# VLA-RL: Towards Masterful and General Robotic Manipulation with Scalable Reinforcement Learning

<div align="center">

[![blog](https://img.shields.io/badge/Notion-000000?style=for-the-badge&logo=notion&logoColor=white)](https://congruous-farmhouse-8db.notion.site/VLA-RL-Toward-Masterful-and-General-Robotic-Manipulation-with-Scalable-Reinforcement-Learning-1953a2cd706280ecaad4e93a5bd2b8e3?pvs=4)

</div>

## 🌟 Highlights

- 🎯 **General Manipulation**: Improving OpenVLA-7B with outcome-based multi-task reinforcement learing.

- ⚡️ **Cutting-edge Architecture**: Built with Ray+vLLM+LoRA+FSDP, our codebase delivers both scalability and flexibility.

- 📝 **Clean Implementation**: Following [cleanrl](https://github.com/vwxyzjn/cleanrl)'s philosophy, we provide a single-file implementation for easy reading and modification.

- 🚧 **Active Development**: Work in Progress, let's build it together.

## 📝 TODO
- [ ] Support SERL-style Real-world RL
- [ ] Support More Environments (e.g., Roboverse)
- [ ] Support More VLAs (e.g., MiniVLA)

## 🛠️ Installation

See [INSTALL.md](docs/INSTALL.md) for installation instructions. 

See [ERROR_CATCH.md](docs/ERROR_CATCH.md) for error catching.

## 🚀 Quick Start

Before launching distributed training, please edit the script with the appropriate dataset and model paths first.

### 📈 Training

For PPO:
```bash
# bash scripts/train_rl_vllm_ray_fsdp.sh <gpus> <task_ids>
# e.g., 
bash scripts/train_rl_vllm_ray_fsdp.sh 0,1 0,1,2,3,4,5,6,7,8,9
```

For GRPO:
```bash
# bash scripts/train_grpo_vllm_ray_fsdp.sh <gpus> <task_ids>
# e.g., 
bash scripts/train_grpo_vllm_ray_fsdp.sh 0,1 0,1,2,3,4,5,6,7,8,9
```

### 🧪 Evaluation

```bash
# parallel evaluation with vectorized environment
bash scripts/eval_vllm_ray.sh 0,1
```

## 🏷️ License

This repository is released under the Apache-2.0 license.

## 🙏 Acknowledgement

Our code is built upon [open-instruct](https://github.com/allenai/open-instruct), [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), [verl](https://github.com/volcengine/verl) and [openvla](https://github.com/openvla/openvla). We thank all these authors for their nicely open sourced code and their great contributions to the community.

## 🥰 Citation

If you find this repository helpful, please consider citing:

```
@misc{lu2025vlarl,
  title={VLA-RL: Towards Masterful and General Robotic Manipulation with Scalable Reinforcement Learning},
  author={Guanxing Lu, Chubin Zhang, Haonan Jiang, Yuheng Zhou, Zifeng Gao, Yansong Tang and Ziwei Wang},
  year={2025},
  howpublished={\url{https://congruous-farmhouse-8db.notion.site/VLA-RL-Towards-Masterful-and-General-Robotic-Manipulation-with-Scalable-Reinforcement-Learning-1953a2cd706280ecaad4e93a5bd2b8e3?pvs=4}},
  note={Notion Blog}
}
```
