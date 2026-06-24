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

#mistral3
from .svd_mistral_3 import (
    PaluMistral3Config,
    PaluMistral3ForCausalLM
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
from .modules import reorder_linear_weight, reorder_linear_weight_cka_cluster, reorder_linear_weight_based_on_histogram

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
    'mistral3': {
        'config': PaluMistral3Config,
        'ModelForCausalLM': PaluMistral3ForCausalLM
    },
    'qwen2': {
        'config': PaluQwen2Config,
        'ModelForCausalLM': PaluQwen2ForCausalLM
    }
}