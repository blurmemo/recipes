from torch import nn


from recipes.tuners.lora.layer import LinearLoraLayer
from recipes.models.utils import freeze
from recipes.tuners.utils import check_part_module_name, recursive_getattr, recursive_setattr

class LoraModel:
    def __init__(self, config, model):
        super().__init__()
        self.config = config
        self.arch = model
        self.layers_wrapped = []
        self._pipline()

    def _pipline(self):
        self._preprocess()
        self._wrap()

    def _preprocess(self):
        self.arch = freeze(self.arch)

    def _wrap(self):
        for name, module in self.arch.named_modules():
            if isinstance(module, nn.Linear):
                self._wrap_linear_layer(name, self.config.linear)

    def _wrap_linear_layer(self, name, config):
        if check_part_module_name(name, config.target_names):
            self.layers_wrapped.append(name)
            module = recursive_getattr(self.arch, name)
            lora_layer = LinearLoraLayer(module.weight, config.r, config.alpha, config.dropout, module.bias).to(module.weight.device)
            recursive_setattr(self.arch, name, lora_layer)
