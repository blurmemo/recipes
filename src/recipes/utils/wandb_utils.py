from dataclasses import dataclass, asdict




def init_wandb(config, train_config):
    """
    config: wandb config
    """
    try:
        import wandb
    except ImportError:
        raise ImportError(
            "You are trying to use wandb which is not currently installed. "
            "Please install it using pip install wandb"
        )

    run = wandb.init(**asdict(config))
    run.config.update(train_config, allow_val_change=True)
    return run