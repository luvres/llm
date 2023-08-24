# 1) qLoRA (Tokenizer Model)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
print(f'CUDA Avaliable: {torch.cuda.is_available()}')

# Environment variable
USER = os.environ['USER']

# Model name
#model_name = "bloomz-3b"
model_name = "bloomz-7b1"
model_subname = "qlora"

model_id = "/scratch/LLM/BLOOM/{model_name}"
model_pretrained =  f"/scratch/{USER}/adapters/{model_name}-{model_subname}"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")

from peft import prepare_model_for_kbit_training

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

print(model)


# 3) Setting up the LoRA Adapters
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

# Initializing LoRA Config
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["query_key_value"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)

print_trainable_parameters(model)

model.save_pretrained(model_pretrained)
