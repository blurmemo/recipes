from recipes.models.mllama.architecture import MllamaForConditionalGeneration, FastMllamaForConditionalGeneration


def mllama(config):
    model = MllamaForConditionalGeneration.from_pretrained(
        config.model_name,
        attn_implementation=config.fast_kernel,
        torch_dtype=config.model_dtype,
    )
    model.supports_gradient_checkpointing = config.gradient_checkpointing
    model.language_model.supports_gradient_checkpointing = config.gradient_checkpointing
    return model

def fast_mllama(config):
    model = FastMllamaForConditionalGeneration.from_pretrained(
        config.model_name,
        attn_implementation=config.fast_kernel,
        torch_dtype=config.model_dtype,
    )
    model.supports_gradient_checkpointing = config.gradient_checkpointing
    model.language_model.supports_gradient_checkpointing = config.gradient_checkpointing
    return model