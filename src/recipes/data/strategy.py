from recipes.data.collator import DATA_COLLATOR
from recipes.data.sampler import SAMPLER, DISTRIBUTED_SAMPLER
from transformers import default_data_collator


class GenericStrategy:
    @staticmethod
    def generate(strategy, dataset, batch_size, processor, partition="train"):
        kwargs = {}
        kwargs["batch_sampler"] = SAMPLER[strategy](
            dataset, batch_size, shuffle=partition == "train", drop_last=True
        )
        kwargs["collate_fn"] = DATA_COLLATOR[strategy](dataset, processor)
        return kwargs

class PaddingStrategy:
    @staticmethod
    def generate(strategy, dataset, batch_size, processor, partition="train"):
        kwargs = {}
        kwargs["batch_sampler"] = SAMPLER["padding"](
            dataset, batch_size, shuffle=partition == "train", drop_last=True
        )
        kwargs["collate_fn"] = DATA_COLLATOR["padding"](processor)
        return kwargs

class PackingStrategy:
    @staticmethod
    def generate(strategy, dataset, batch_size, processor, partition="train"):
        kwargs = {}
        kwargs["batch_size"] = batch_size
        kwargs["drop_last"] = True if partition == "train" else False
        kwargs["collate_fn"] = default_data_collator
        return kwargs


class MetaStrategy:
    """
    this is your own meta dataset strategy for dataloader kwargs
    """
    pass

STRATEGY = {
    "padding": PaddingStrategy,
    "packing": PackingStrategy
}


class DistributedGenericStrategy(GenericStrategy):
    @staticmethod
    def generate(strategy, dataset, batch_size, processor, partition="train", rank=None, world_size=None):
        kwargs = {}
        kwargs["batch_sampler"] = DISTRIBUTED_SAMPLER[strategy](
            rank, dataset, batch_size,
            num_replicas=world_size,
            shuffle=partition == "train", drop_last=True
        )
        kwargs["collate_fn"] = DATA_COLLATOR[strategy](processor)
        return kwargs


class DistributedPaddingStrategy(PaddingStrategy):
    @staticmethod
    def generate(strategy, dataset, batch_size, processor, partition="train", rank=None, world_size=None):
        kwargs = {}
        kwargs["batch_sampler"] = DISTRIBUTED_SAMPLER["padding"](
            rank, dataset, batch_size,
            num_replicas=world_size,
            shuffle=partition == "train", drop_last=True
        )
        kwargs["collate_fn"] = DATA_COLLATOR["padding"](processor)
        return kwargs

DISTRIBUTED_STRATEGY = {
    "padding": PaddingStrategy,
}