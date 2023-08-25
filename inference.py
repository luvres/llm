import os
import argparse
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

#os.environ["CUDA_VISIBLE_DEVICES"]="0"
print(f'CUDA Avaliable: {torch.cuda.is_available()}')

# Environment variable
USER = os.environ['USER']

parser = argparse.ArgumentParser(description="Prepared Adapter")
parser.add_argument('--model_name', type=str, help="Name of the model, example: 'bloomz-3b'", required=True)
parser.add_argument('--peft_method', type=str, choices={'lora','qlora'}, default='qlora')
#parser.add_argument('--tuning', type=str, choices={'instruction','fine'}, default='fine')
args = parser.parse_args()

model_name = args.model_name
peft_method = args.peft_method

peft_model_id = f"/scratch/{USER}/adapters/{model_name}-{peft_method}"

config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    return_dict=True,
    load_in_8bit=False
)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_id)



def generate_prompt(user: str, chip2: str) -> str:
  prompt = f"### INSTRUCTION\nO primeiro treinamento.\n\n### User:\n{user}\n### Chip2:\n{chip2}"
  return prompt

mapped_dataset = dataset_reduced.map(lambda samples: tokenizer(generate_prompt(samples['user'], samples['chip2'])))






