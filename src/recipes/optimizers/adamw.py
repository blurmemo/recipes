from torch import optim



class AdamWConfig:
    lr: float = 1e-3,
    eps: float = 1e-8,
    weight_decay: float = 1e-2,


def adamw(config: AdamWConfig, model):
    return optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )