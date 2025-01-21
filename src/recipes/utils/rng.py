from dataclasses import dataclass
import random
import numpy as np
import torch

@dataclass
class Rng:
    random_rng_state = random.getstate()
    np_rng_state = np.random.get_state()
    torch_rng_state = torch.get_rng_state()
    torch_cuda_rng_state = torch.cuda.get_rng_state()