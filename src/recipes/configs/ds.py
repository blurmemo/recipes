
DSConfig = {
    "max_train_batch_size": None,  # int
    "train_micro_batch_size_per_gpu": 1,  # int
    "train_batch_size": 2,  # int   train_batch_size = train_micro_batch_size * num gpus * gradient_accumulation_steps
    # gradient
    "gradient_accumulation_steps": 1,  #int
    "gradient_clipping": 1.0,  # float
    "prescale_gradients": False,  # bool
    "fp16": {
        "enable": False,  # bool
        "loss_scale_window": 100,  # int
    },
    "bf16": {
        "enable": False,
    },
    "zero_optimization": {
        "stage": 0,  # int
        "offload_param": {
            "device": "cuda",
        },
        "offload_optimizer": {
            "device": "none",
        }
    },
    "steps_per_print": 100000,  # int
    "wall_clock_breakdown": False,  # bool
}




















