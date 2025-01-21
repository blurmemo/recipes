from dataclasses import dataclass
from recipes.utils.rng import Rng



@dataclass
class CheckpointConfig:
    rng: Rng = Rng()
    step: int = 0
    batch_size: int = 1
    loss: float = 0.0
    best_loss: float = float("inf")
    model = None
    optimizer = None
    scheduler = None

