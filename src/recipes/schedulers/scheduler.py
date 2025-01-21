from recipes.schedulers.steplr import steplr
from recipes.schedulers.steplr import StepLRConfig

SCHEDULER = {
    StepLRConfig: steplr,
}


class Scheduler:
    @staticmethod
    def build(config, optimizer):
        scheduler = SCHEDULER[type(config)]
        return scheduler(config, optimizer)