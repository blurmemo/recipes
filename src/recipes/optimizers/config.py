from dataclasses import dataclass, field

from recipes.optimizers.adamw import AdamWConfig



@dataclass
class OptimizerConfig:
    optimizers: dict = field(default_factory=lambda: {
        "adamw": AdamWConfig,
    })

    @staticmethod
    def generate(name: str = None):
        config = OptimizerConfig()
        names = tuple(config.optimizers.keys())
        assert name in names, f"optimizer: {name} is not implemented."
        return config.optimizers[name]()
