# Usage:
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 torchrun --nproc_per_node=6 test/test_broadcast.py

import os
import torch
import torch.distributed as dist
import deepspeed
import numpy as np


def setup_distributed():
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # if not dist.is_initialized():
    # dist.init_process_group(backend="nccl")
    deepspeed.init_distributed()
    torch.cuda.set_device(rank)
    return rank, world_size

def test_broadcast():
    rank, world_size = setup_distributed()
    
    # SHAPE = (10, 6, 224, 224)
    # if rank == 0:
    #     x = torch.randn(SHAPE).cuda()
    # else:
    #     x = torch.empty(SHAPE).cuda()

    # x = np.empty((1, 1), dtype=np.object_)
    # if rank == 0:
    #     x[0, 0] = "Hello, world"
    
    if rank == 0:
        x = "Hello, world"
    else:
        x = ""
    x = [x]
    # dist.broadcast(x, src=0)
    dist.broadcast_object_list(x, src=0)

    # dist.barrier()
    
    # check by print
    rank = dist.get_rank()
    # print(f"Tensor on rank {rank} after broadcast: {x[0, 0, 0, 0]}")
    print(f"Tensor on rank {rank} after broadcast: {x}")
    
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    test_broadcast()
