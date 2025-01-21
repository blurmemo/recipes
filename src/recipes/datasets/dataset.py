from torch.utils.data import Dataset


class MetaDataset(Dataset):
    def __init__(self, config, processor, partition="train"):
        self.config = config
        self.processor = processor
        self.partition = partition

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

