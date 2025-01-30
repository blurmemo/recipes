from dataclasses import dataclass, field
import torch
from torch.distributed.fsdp import ShardingStrategy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer


@dataclass
class FSDPConfig:
    mixed_precision: bool = True
    model_dtype: torch.dtype = torch.float16
    transformer_layers: set = field(default_factory=lambda: {})   # you need set

    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD # HYBRID_SHARD "Full Shard within a node DDP cross Nodes", SHARD_GRAD_OP "Shard only Gradients and Optimizer States", NO_SHARD "Similar to DDP".
    # device mesh
    hsdp: bool = False # Require HYBRID_SHARD to be set. This flag can extend the HYBRID_SHARD by allowing sharding a model on customized number of GPUs (Sharding_group) and Replicas over Sharding_group.
    sharding_group_size: int = 0 # requires hsdp to be set. This specifies the sharding group size, number of GPUs that you model can fit into to form a replica of a model.
    replica_group_size: int = 0 #requires hsdp to be set. This specifies the replica group size, which is world_size/sharding_group_size.

    activate_checkpoint: bool = True
    checkpoint_submodule = LlamaDecoderLayer

    cpu_offload: bool = False
    low_cpu: bool = False

    # if freeze True else False
    use_orig_params: bool = True