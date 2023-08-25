import os
import argparse
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

#os.environ["CUDA_VISIBLE_DEVICES"]="0"
print(f'CUDA Avaliable: {torch.cuda.is_available()}')

# Environment variable
USER = os.environ['USER']

parser = argparse.ArgumentParser(description="Prepared Adapter")
parser.add_argument('--model_name', type=str, help="Name of the model, example: 'bloomz-7b1'", required=True)
parser.add_argument('--peft_method', type=str, choices={'lora','qlora'}, default='qlora')
#parser.add_argument('--tuning', type=str, choices={'instruction','fine'}, default='fine')
args = parser.parse_args()

model_name = args.model_name
peft_method = args.peft_method

peft_model_id = f"/scratch/{USER}/adapters/{model_name}-{peft_method}"
#peft_model_id = f"/scratch/{USER}/adapters/bloomz-7b1-lora"

config = PeftConfig.from_pretrained(peft_model_id)

model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    return_dict=True,
    load_in_8bit=False
)

tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_id)


# Inference
#def make_inference(user: str) -> str:
def make_inference():
  batch = tokenizer(f"Quais são as estações do ano?.", return_tensors='pt')
  with torch.cuda.amp.autocast():
    output_tokens = model.generate(**batch, max_new_tokens=50)
    
  print('\n\n', tokenizer.decode(output_tokens[0], skip_special_tokens=True))

#mapped_dataset = dataset_reduced.map(lambda samples: tokenizer(generate_prompt(samples['user'], samples['chip2'])))
#user_here = "Clima"
#chip2_here = ""

#make_inference(user_here)
make_inference()




