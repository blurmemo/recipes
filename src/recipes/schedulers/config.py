from dataclasses import field, dataclass, asdict

from recipes.schedulers.steplr import StepLRConfig
from recipes.utils.utils import update_attrdict

@dataclass
class SchedulerConfig:
    schedulers: dict = field(default_factory=lambda: {
        "steplr": StepLRConfig
    })

    @staticmethod
    def generate(config):
        name = config.scheduler
        scheduler_config = SchedulerConfig()
        names = tuple(scheduler_config.schedulers.keys())
        assert name in names, f"scheduler: {name} is not implemented."
        sc = scheduler_config.schedulers[name]()
        update_attrdict(sc, asdict(config))
        return sc