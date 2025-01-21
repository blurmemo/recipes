from torch import nn
from transformers import AutoConfig

from recipes.models.zoo import ZOO


class Model(nn.Module):
    def __init__(self, config, tokenizer):
        """
        config: train config
        tokenizer: tokenizer
        """
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.model = None
        self._pipline()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def _pipline(self):
        self._build()


    def _build(self):
        model_config = AutoConfig.from_pretrained(self.config.model_name)
        self.model = ZOO[model_config.model_type](self.config)

        if len(self.tokenizer) > self.model.get_input_embeddings().weight.shape[0]:
            print("WARNING: Resize the embedding matrix to match the tokenizer vocab size.")
            self.model.resize_token_embeddings(len(self.tokenizer))



