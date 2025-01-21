from functools import partial

from recipes.datasets.config import MetaConfig
from recipes.datasets.dataset import MetaDataset


DATASET = {
    "meta": MetaDataset,
}



CONFIG = {
    "meta": MetaConfig,
}

__all__ = ["DATASET", "CONFIG"]
