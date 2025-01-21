from torch.optim.lr_scheduler import StepLR


class StepLRConfig:
    step_size: int = 1
    gamma: float = 0.85


def steplr(config: StepLRConfig, optimizer):
    print(config)
    step_size = config.step_size
    gamma = config.gamma
    return StepLR(optimizer, step_size=step_size, gamma=gamma)