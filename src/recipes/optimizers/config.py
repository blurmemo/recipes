from dataclasses import dataclass, field, asdict

from recipes.optimizers.adamw import AdamWConfig
from recipes.utils.utils import update_attrdict


@dataclass
class OptimizerConfig:
    optimizers: dict = field(default_factory=lambda: {
        "adamw": AdamWConfig,
    })

    @staticmethod
    def generate(config):
        """
        config: train config
        """
        name = config.optimizer
        optimizer_config = OptimizerConfig()
        names = tuple(optimizer_config.optimizers.keys())
        assert name in names, f"optimizer: {name} is not implemented."
        oc = optimizer_config.optimizers[name]()
        update_attrdict(oc, asdict(config))
        return oc
