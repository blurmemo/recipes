from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

class Wrapper:
    def __init__(self, model, rank=None):
        super().__init__()
        self.arch = model
        self.rank = rank
        self._pipline()

    def _pipline(self):
        self._wrap()

    def _wrap(self):
        self.arch = DDP(self.arch, device_ids=[self.rank], find_unused_parameters=True)