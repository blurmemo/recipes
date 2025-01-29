from dataclasses import dataclass, asdict
import wandb


class Wandb:
    @staticmethod
    def generate(wandb_config, configs: list, output_dir: str = "./"):
        run = wandb.init(**asdict(wandb_config), dir=output_dir)
        for config in configs:
            run.config.update({type(config).__name__: asdict(config)})
        return run
