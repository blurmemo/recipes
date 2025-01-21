from recipes.optimizers.adamw import AdamWConfig, adamw


OPTIMIZER = {
    AdamWConfig: adamw,
}


class Optimizer:
    @staticmethod
    def build(config, model):
        optimizer = OPTIMIZER[type(config)]
        return optimizer(config, model)