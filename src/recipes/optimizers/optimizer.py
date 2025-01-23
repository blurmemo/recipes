from recipes.optimizers.adamw import AdamWConfig, adamw


OPTIMIZER = {
    AdamWConfig: adamw,
}


class Optimizer:
    @staticmethod
    def build(config, model):
        """
        config: optimizer config such as AdamWConfig
        model: model to optimize
        """
        optimizer = OPTIMIZER[type(config)]
        return optimizer(config, model)
