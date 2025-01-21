from transformers import AutoProcessor, AutoTokenizer


class Processor:
    def __init__(self, config):
        self.config = config
        self.processor = None
        self.tokenizer = None
        self._pipline()

    @staticmethod
    def build(config):
        mix_processor = Processor(config)
        return mix_processor.processor, mix_processor.tokenizer

    def _pipline(self):
        if self.config.is_vision:
            self._vision()
        else:
            self._language()

    def _vision(self):
        data_processor = AutoProcessor.from_pretrained(self.config.model_name)
        data_processor.tokenizer.padding_side = 'right'
        self.processor = data_processor
        self.tokenizer = data_processor.tokenizer

    def _language(self):
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if not tokenizer.pad_token_id:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        self.processor = tokenizer
        self.tokenizer = tokenizer






