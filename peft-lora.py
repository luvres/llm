# 1) LoRA (Tokenizer Model)
import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
print(f'CUDA Avaliable: {torch.cuda.is_available()}')

# Environment variable
USER = os.environ['USER']
#
# Model name
#model_name = "bloomz-3b"
model_name = "bloomz-7b1"
############
#
model_id = "/scratch/LLM/BLOOM/{model_name}"
model_pretrained =  f"/scratch/{USER}/{model_name}-adapter-qlora"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    load_in_8bit=True,
    device_map='auto',
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

print(model)


# 2) Freezing the original wheigths 
for param in model.parameters():
  param.requires_grad = False  # freeze the model - train adapters later
  if param.ndim == 1:
    # cast the small parameters (e.g. layernorm) to fp32 for stability
    param.data = param.data.to(torch.float32)

model.gradient_checkpointing_enable()  # reduce number of stored activations
model.enable_input_require_grads()

class CastOutputToFloat(nn.Sequential):
  def forward(self, x): return super().forward(x).to(torch.float32)
model.lm_head = CastOutputToFloat(model.lm_head)


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
