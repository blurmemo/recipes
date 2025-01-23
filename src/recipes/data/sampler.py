import random
from itertools import islice

import numpy as np
import torch


class LengthBasedBatchSampler(torch.utils.data.BatchSampler):
    def __init__(self, dataset, batch_size: int, shuffle: bool=True, drop_last: bool=True) -> None:
        if isinstance(next(iter(dataset)), dict):
            first_key = next(iter(next(iter(dataset)).keys()))
            self.lengths = [len(d[first_key]) for d in dataset]
        else:
            self.lengths = [len(d) for d in dataset]
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

    def __iter__(self):
        ids = np.argsort(self.lengths, kind='mergesort')
        if self.drop_last:
            ids = ids[:len(ids) // self.batch_size * self.batch_size]

        batches = [ids[i:i+self.batch_size] for i in range(0, len(ids), self.batch_size)]

        if self.shuffle:
            random.shuffle(batches)

        for b in batches:
            yield b

    def __len__(self):
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        else:
            return len(self.lengths) // self.batch_size + (len(self.lengths) % self.batch_size > 0)


class DistributedLengthBasedBatchSampler(torch.utils.data.BatchSampler):
    def __init__(self, rank: int, dataset, batch_size: int, num_replicas: int, shuffle: bool = True, drop_last: bool=True, seed: int = 0) -> None:
        random.seed(seed)
        self.rank = rank
        self.batch_sampler = LengthBasedBatchSampler(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
        self.num_replicas = num_replicas

    def __iter__(self):
        max_length = len(self.batch_sampler) // self.num_replicas * self.num_replicas
        return islice(self.batch_sampler, self.rank, max_length, self.num_replicas)

    def __len__(self):
        return len(self.batch_sampler) // self.num_replicas


SAMPLER = {
    "padding": LengthBasedBatchSampler,
}

DISTRIBUTED_SAMPLER = {
    "padding": DistributedLengthBasedBatchSampler,
}