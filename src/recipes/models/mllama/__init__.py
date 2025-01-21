from typing import TYPE_CHECKING

from transformers.utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
    is_vision_available,
)


_import_structure = {
    "configuration_mllama": ["MllamaConfig"],
    "processing_mllama": ["MllamaProcessor"],
}


try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["architecture"] = [
        "FastMllamaForConditionalGeneration"
        "MllamaForConditionalGeneration",
        "MllamaForCausalLM",
        "MllamaTextModel",
        "MllamaVisionModel",
        "MllamaPreTrainedModel",
    ]

try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["image_processing_mllama"] = ["MllamaImageProcessor"]


if TYPE_CHECKING:
    from .configuration_mllama import MllamaConfig
    from .processing_mllama import MllamaProcessor

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .architecture import (
            MllamaForCausalLM,
            FastMllamaForConditionalGeneration,
            MllamaForConditionalGeneration,
            MllamaPreTrainedModel,
            MllamaTextModel,
            MllamaVisionModel,
        )

    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .image_processing_mllama import (
            MllamaImageProcessor,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
