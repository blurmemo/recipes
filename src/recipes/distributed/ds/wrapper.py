import deepspeed
from dataclasses import dataclass, asdict


class Wrapper:
    def __init__(self, config, model, optimizer, scheduler, rank=None):
        self.config = config
        self.arch = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self._pipline()

    def _pipline(self):
        self._wrap()

    def _wrap(self):
        self.arch, self.optimizer, _, self.scheduler = deepspeed.initialize(
            model=self.arch,
            optimizer=self.optimizer,
            lr_scheduler=self.scheduler,
            model_parameters=self.arch.parameters(),
            config=self.config if type(self.config) is dict else asdict(self.config),
        )