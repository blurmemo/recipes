from recipes.datasets import CONFIG, DATASET
from torch.utils.data import Dataset


def generate_dataset_config(name: str = None):
    names = tuple(CONFIG.keys())
    assert name in names, f"unknown config: {name}"
    config = CONFIG[name]()
    return config

def generate_dataset(config, processor, partition: str = "train") -> Dataset:
    if config.dataset not in DATASET:
        raise NotImplementedError(f"dataset: {config.dataset} is not implemented")
    dataset = DATASET[config.dataset](config, processor, partition)
    return dataset