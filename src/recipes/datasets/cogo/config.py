from dataclasses import dataclass
from recipes.datasets.config import MetaConfig

@dataclass
class CogoConfig(MetaConfig):
    dataset: str = "cogo"
    images: str = "/data0/home/ening/NICA/cogmllm/datasets/cogo/images"
    annotations: str = "/data0/home/ening/NICA/cogmllm/datasets/cogo/annotations"
    train: str = "cogo_train.json"
    validation: str = "cogo_val.json"
    test: str = None
    prompt: str = "cogo_prompt.json"

@dataclass
class BenchmarkConfig(CogoConfig):
    test: str = "cogo_test.json"