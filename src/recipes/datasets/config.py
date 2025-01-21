from dataclasses import dataclass


@dataclass
class MetaConfig:
    dataset: str = "meta"
    annotations: str = "PATH/annotations_dir"
    train: str = "train dataset filename"
    validation: str = "validation dataset filename"
    test: str = "test dataset filename"

