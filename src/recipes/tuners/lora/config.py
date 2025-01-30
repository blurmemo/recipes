from dataclasses import dataclass, field
from typing import List
import torch
from torch import nn

from recipes.tuners.config import TunerConfig


@dataclass
class LinearConfig(TunerConfig):
     r: int = 8
     target_names: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
     alpha: int = 32
     bias = "none"
     dropout: float = 0.05


@dataclass
class LoraConfig:
     linear: LinearConfig = LinearConfig()