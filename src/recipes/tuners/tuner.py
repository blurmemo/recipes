import torch
from torch import nn
from recipes.tuners.lora.config import LoraConfig
from recipes.tuners.lora.model import LoraModel


TUNER_MODEL = {
    LoraConfig: LoraModel,
}


class Tuner(nn.Module):
    def __init__(self, config, model, **kwargs):
        super().__init__()
        self.config = config
        self.model = model
        self._pipline()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def _pipline(self):
        self._convert()

    def _convert(self):
        tuner_model = TUNER_MODEL[type(self.config)]
        self.model = tuner_model(self.config, self.model)



