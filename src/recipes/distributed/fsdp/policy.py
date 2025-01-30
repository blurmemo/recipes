import torch
import torch.distributed as dist
import torch.cuda.nccl as nccl
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.wrap import _or_policy, lambda_auto_wrap_policy, transformer_auto_wrap_policy
import functools

class Policy:
    def __init__(self, config, rank=None):
        self.config = config
        self.rank = rank
        self._bf = None
        self.mixed_precision = None
        self.auto_wrap = None
        self._pipline()

    def _preprocess(self):
        self._bf = (
                torch.version.cuda and
                torch.cuda.is_bf16_supported() and
                torch.version.cuda >= "11.0" and
                dist.is_nccl_available() and
                nccl.version() >= (2, 10)
        )
        self._fpSixteen = MixedPrecision(
            param_dtype=torch.float16,
            # Gradient communication precision.
            reduce_dtype=torch.float16,
            # Buffer precision.
            buffer_dtype=torch.float16,
        )

        self._bfSixteen = MixedPrecision(
            param_dtype=torch.bfloat16,
            # Gradient communication precision.
            reduce_dtype=torch.bfloat16,
            # Buffer precision.
            buffer_dtype=torch.bfloat16,
            cast_forward_inputs=True,
        )
        self._bfSixteen_mixed = MixedPrecision(
            param_dtype=torch.float32,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
        self._fp32 = MixedPrecision(
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        )

    def _pipline(self):
        self._preprocess()
        self._build()

    def _build_mixed_precision(self):
        use_fp16 = self.config.model_dtype is torch.float16
        if self.config.mixed_precision:
            if self._bf and not use_fp16:
                self.mixed_precision = self._bfSixteen
            elif use_fp16:
                self.mixed_precision = self._fpSixteen

    def _build_wrap(self):
        lambda_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_fn)
        transformer_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=self.config.transformer_layers,
        )

        auto_wrap_policy = functools.partial(_or_policy, policies=[lambda_policy, transformer_wrap_policy])
        self.auto_wrap = auto_wrap_policy

    def _build(self):
        self._build_mixed_precision()
        self._build_wrap()

def lambda_fn(module):
    if (
        len(list(module.named_children())) == 0 and
        getattr(module, "weight", None) is not None and
        module.weight.requires_grad
    ):
        return True
    return False