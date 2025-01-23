import random
import torch
import numpy as np



def set_seed(seed: int = 0):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def update_attrdict(obj, attrdict: dict):
    for k, v in attrdict.items():
        if hasattr(obj, k):
            setattr(obj, k, v)
