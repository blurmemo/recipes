from dataclasses import dataclass
from recipes.utils.rng import Rng
import torch


@dataclass
class CheckpointConfig:

    def __init__(self, step: int = 0, batch_size: int = 1, loss: float = 0.0, best_loss: float = float("inf"), model=None, optimizer=None, scheduler=None):
        self.rng = Rng()
        self.step = step
        self.batch_size = batch_size
        self.loss = loss
        self.best_loss = best_loss
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

