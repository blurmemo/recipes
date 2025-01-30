from torch.distributed._tensor import init_device_mesh
import os
import torch.distributed as dist
from torch.distributed.fsdp import StateDictType


def check_config(config):
    VALID_TYPES = (StateDictType.SHARDED_STATE_DICT, StateDictType.FULL_STATE_DICT)
    if isinstance(config.checkpoint_type, str):
        str_to_obj = {
            "StateDictType.SHARDED_STATE_DICT": StateDictType.SHARDED_STATE_DICT,
            "StateDictType.FULL_STATE_DICT": StateDictType.FULL_STATE_DICT,
        }
        if config.checkpoint_type in str_to_obj:
            config.checkpoint_type = str_to_obj[config.checkpoint_type]

    if not config.checkpoint_type in VALID_TYPES:
        raise ValueError(f"Invalid checkpoint_type {config.checkpoint_type}")


def hsdp_device_mesh(replica_group_size, sharding_group_size, device=None):
    if replica_group_size is None or sharding_group_size is None:
        raise ValueError("Both replica_group_size and sharding_group_size must be provided.")

    world_size = dist.get_world_size()

    device = device or f"cuda"

    if world_size % sharding_group_size != 0:
        raise ValueError(f"World size {world_size} is not evenly divisible by "
                         f"sharding group size {sharding_group_size}.")

    if (world_size // sharding_group_size) % replica_group_size != 0:
        raise ValueError(f"The calculated number of replica groups is not evenly divisible by "
                         f"replica_group_size {replica_group_size}.")

    device_mesh = init_device_mesh(device, (replica_group_size, sharding_group_size))
    if device_mesh is None:
        raise RuntimeError("Failed to create a valid device mesh.")
    return device_mesh