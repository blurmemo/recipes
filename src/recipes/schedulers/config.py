from dataclasses import field, dataclass

from recipes.schedulers.steplr import StepLRConfig

@dataclass
class SchedulerConfig:
    schedulers: dict = field(default_factory=lambda: {
        "steplr": StepLRConfig
    })

    @staticmethod
    def generate(name: str = None):
        config = SchedulerConfig()
        names = tuple(config.schedulers.keys())
        assert name in names, f"scheduler: {name} is not implemented."
        return config.schedulers[name]()