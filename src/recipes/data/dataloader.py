import torch

from recipes.datasets.utils import generate_dataset_config, generate_dataset
from recipes.data.strategy import STRATEGY, DISTRIBUTED_STRATEGY

class DataLoader:
    def __init__(self, config, processor, tokenizer, batch_size, partition="train"):
        """
        config: training config
        processor: data processor
        partition: train or val
        """
        self.config = config
        self.processor = processor
        self.tokenizer = tokenizer
        self.partition = partition
        self.batch_size = batch_size
        self._pipline()

    def _pipline(self):
        self._preprocess()
        self._build()

    def _preprocess(self):
        self.dataset_config = generate_dataset_config(self.config.dataset)


    def _build(self):
        dataset = generate_dataset(self.dataset_config, self.processor, self.partition)
        self.dataset = dataset
        print(f"--> {self.partition} dataset length = {len(dataset)}")
        self.strategy_kwargs = STRATEGY[self.config.batch_strategy].generate(
            self.config.batch_strategy, dataset, self.batch_size, self.processor, self.tokenizer, self.partition
        )
        self.dataloader = torch.utils.data.DataLoader(
            dataset, num_workers=self.config.num_workers, pin_memory=True, **self.strategy_kwargs
        )
        print(f"--> {self.partition} dataloader batches length = {len(self.dataloader)}")

    def __len__(self):
        return len(self.dataloader)

    def sample(self):
        epoch = self.config.epoch
        length = len(self.dataloader)
        for _ in range(epoch):
            for step, batch in enumerate(self.dataloader):
                yield (epoch - 1) * length + step, batch

class DistributedDataLoader(DataLoader):
    def __init__(self, config, processor, tokenizer, batch_size, partition="train", rank=None, world_size=None):
        self.rank = rank
        self.world_size = world_size
        super().__init__(config, processor, tokenizer, batch_size, partition)

    def _build(self):
        dataset = generate_dataset(self.dataset_config, self.processor, self.partition)
        self.dataset = dataset
        print(f"--> {self.partition} dataset length = {len(dataset)}")
        self.strategy_kwargs = DISTRIBUTED_STRATEGY[self.config.batch_strategy].generate(
            self.config.batch_strategy, dataset, self.batch_size, self.processor, self.tokenizer, self.partition,
            rank=self.rank, world_size=self.world_size
        )
        self.dataloader = torch.utils.data.DataLoader(
            dataset, num_workers=self.config.num_workers, pin_memory=True, **self.strategy_kwargs,
        )
        print(f"--> {self.partition} dataloader batches length = {len(self.dataloader)}")
