from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

class Wrapper(nn.Module):
    def __init__(self, model, rank=None):
        super().__init__()
        self.model = model
        self.rank = rank
        self._pipline()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def _pipline(self):
        self._wrap()

    def _wrap(self):
        self.model = DDP(self.model, device_ids=[self.rank], find_unused_parameters=True)