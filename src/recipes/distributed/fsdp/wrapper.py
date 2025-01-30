import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from recipes.distributed.fsdp.utils import check_config, hsdp_device_mesh
from recipes.distributed.fsdp.policy import Policy
from functools import partial
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper, CheckpointImpl, apply_activation_checkpointing



class Wrapper:
    def __init__(self, config, model, rank=None):
        self.config = config
        self.rank = rank
        self.arch = model
        self.device_mesh = None
        self.policy = None
        self._pipline()

    def _preprocess(self):
        check_config(self.config)
        if self.config.hsdp and self.config.sharding_strategy == ShardingStrategy.HYBRID_SHARD:
            device_mesh = hsdp_device_mesh(
                replica_group_size=self.config.replica_group_size,
                sharding_group_size=self.config.sharding_group_size,
            )
            self.device_mesh = device_mesh
        self.policy = Policy(self.config)

    def _pipline(self):
        self._preprocess()
        self._wrap()
        self._activate_checkpoint()


    def _wrap(self):
        self.arch = FSDP(
            self.arch,
            auto_wrap_policy=self.policy.auto_wrap,
            cpu_offload=CPUOffload(offload_params=True) if self.config.cpu_offload else None,
            mixed_precision=self.policy.mixed_precision,
            sharding_strategy=self.config.sharding_strategy,
            device_mesh=self.device_mesh,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
            sync_module_states=self.config.low_cpu,
            param_init_fn=(
                lambda module: module.to_empty(
                    device=torch.device("cuda"), recurse=False
                )) if self.config.low_cpu and self.rank != 0 else None,
            use_orig_params=self.config.use_orig_params,
        )


    def _activate_checkpoint(self):
        if self.config.activate_checkpoint:
            self.arch.enable_input_require_grads()
            self.arch.gradient_checkpointing_enable()
            apply_activation_checkpointing(
                self.arch,
                checkpoint_wrapper_fn=partial(
                    checkpoint_wrapper,
                    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                ),
                check_fn=lambda submodule: submodule is self.config.checkpoint_submodule
            )

