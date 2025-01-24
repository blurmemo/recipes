from dataclasses import dataclass, asdict
import wandb


class Wandb:
    @staticmethod
    def generate(wandb_config, configs: list):
        run = wandb.init(**asdict(wandb_config))
        for config in configs:
            run.config.update(config)
        return run
