import torch
from dataclasses import dataclass

@dataclass
class TrainConfig:
    seed: int = 42
    device_map: str = "cuda:1"

    # model config
    model_name: str = "/data0/home/ening/NICA/cogmllm/models/llama/llama_vision_11B_instruct/hf"
    fast_kernel: str = None  # sdpa
    model_dtype: torch.dtype = torch.float16
    gradient_checkpointing: bool = True

    # processor config
    is_vision: bool = True


    # dataset
    dataset: str = "meta"
    train_batch_size: int = 1
    val_batch_size: int = 1
    batch_strategy: str = "meta"  # eq=dataset or padding or packing
    num_workers: int = 1

    # epoch
    epoch: int = 1
    stop_steps: int = 100000
    eval_interval: int = 10000
    eval_stop_steps: int = 10000

    # amp
    amp: bool = False

    # gradient
    gradient_accumulation_steps: int = 1
    gradient_clip: bool = True
    gradient_clip_norm: float = 0.0


    # optimizer
    optimizer: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 0.0

    # scheduler
    scheduler: str = "steplr"
    step_size: int = 1
    gamma: float = 0.85

    # save
    output_dir: str = ""
    profile_mode: str = None # flop or profile
    flop_counter_start: int = 3


