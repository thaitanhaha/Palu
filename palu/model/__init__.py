#llama
from .svd_llama import (
    PaluLlamaConfig,
    PaluLlamaForCausalLM
)

#mistral
from .svd_mistral import (
    PaluMistralConfig,
    PaluMistralForCausalLM
)

#qwen
from .svd_qwen import (
    PaluQwen2Config,
    PaluQwen2ForCausalLM
)

#modules
from .modules import (
    HeadwiseLowRankModule
)
from .modules import reorder_linear_weight, reorder_linear_weight_based_on_histogram

#TODO Mistral



AVAILABLE_MODELS = {
    'llama': {
        'config': PaluLlamaConfig,
        'ModelForCausalLM': PaluLlamaForCausalLM
    },
    'mistral': {
        'config': PaluMistralConfig,
        'ModelForCausalLM': PaluMistralForCausalLM
    },
    'qwen2': {
        'config': PaluQwen2Config,
        'ModelForCausalLM': PaluQwen2ForCausalLM
    }
}