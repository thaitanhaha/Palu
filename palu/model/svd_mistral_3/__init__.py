from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from .configuration_palu_mistral_3 import PaluMistral3Config
from .modeling_palu_mistral_3 import PaluMistral3ForCausalLM

AutoConfig.register("palumistral3", PaluMistral3Config)
AutoModelForCausalLM.register(PaluMistral3Config, PaluMistral3ForCausalLM)
AutoTokenizer.register(PaluMistral3Config, LlamaTokenizer)

