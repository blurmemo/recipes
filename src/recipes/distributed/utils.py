import os
import torch

import torch.distributed as dist


def init_distributed():
    dist.init_process_group("nccl")


def cleanup():
    dist.destroy_process_group()


def get_dist_info():
    local_rank = dist.get_node_local_rank()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return local_rank, rank, world_size


def setup_gpu(local_rank, rank):
    torch.cuda.set_device(local_rank)
    if rank == 0:
        print(f"setup gpu, including clear gpu cache and set environment flags for debugging puposes")
    torch.cuda.empty_cache()
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'


def get_all_reduce_mean(tensor):
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor / dist.get_world_size()

def barrier():
    dist.barrier()