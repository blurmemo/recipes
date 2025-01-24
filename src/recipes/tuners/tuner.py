import torch
from torch import nn
from recipes.tuners.lora.config import LoraConfig
from recipes.tuners.lora.model import LoraModel


TUNER_MODEL = {
    LoraConfig: LoraModel,
}


class Tuner:
    def __init__(self, config, model, **kwargs):
        """
        config: tuner config
        """
        super().__init__()
        self.config = config
        self.arch = model
        self._pipline()

    def _pipline(self):
        self._convert()

    def _convert(self):
        tuner_model = TUNER_MODEL[type(self.config)]
        self.arch = tuner_model(self.config, self.arch)



