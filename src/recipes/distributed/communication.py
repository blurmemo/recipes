import torch
import torch.distributed as dist
import deepspeed

class Communication:
    def __init__(self, engine: str = "default"):
        self.engine = engine

    def init_distributed(self):
        if self.engine == "default":
            dist.init_process_group("nccl")
        elif self.engine == "deepspeed":
            deepspeed.init_distributed("nccl")
        else:
            raise ValueError(f"unsupported accelerator engine {self.engine}")


    def get_all_reduce_mean(self, tensor):
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        tensor = tensor / torch.distributed.get_world_size()
        return tensor