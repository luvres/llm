import os
import argparse
import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

#os.environ["CUDA_VISIBLE_DEVICES"]="0"
print(f'CUDA Avaliable: {torch.cuda.is_available()}')

# Environment variable
USER = os.environ['USER']

parser = argparse.ArgumentParser(description="Prepared Adapter")
parser.add_argument('--model_name', type=str, help="Name of the model, example: 'bloomz-3b'", required=True)
parser.add_argument('--peft_method', type=str, choices={'lora','qlora'}, default='qlora')
parser.add_argument('--lora_r', type=int, default=16)
parser.add_argument('--lora_alpha', type=int, default=32)
parser.add_argument('--lora_target_modules', type=str, default='query_key_value') 
parser.add_argument('--lora_dropout', type=float, default=0.05) # 
parser.add_argument('--lora_bias', type=str, choices={'all','none'}, required='none')
parser.add_argument('--lora_task_type', type=str, default='CAUSAL_LM')

args = parser.parse_args()

model_name = args.model_name
model_subname = args.peft_method
lora_r = args.lora_r
lora_alpha = args.lora_alpha
lora_target_modules = args.lora_target_modules
lora_dropout = args.lora_dropout
lora_bias = args.lora_bias
lora_task_type = args.lora_task_type

model_id = f"/scratch/LLM/BLOOM/{model_name}"
model_pretrained =  f"/scratch/{USER}/adapters/{model_name}-{model_subname}"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    target_modules=[lora_target_modules],
    lora_dropout=lora_dropout,
    bias=lora_bias,
    task_type=lora_task_type
)

#tokenizer = AutoTokenizer.from_pretrained(model_id)
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, config)
model.save_pretrained(model_pretrained)

print(model)

# List trainable params
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

print_trainable_parameters(model)

