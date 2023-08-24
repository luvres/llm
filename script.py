import transformers
from datasets import load_dataset

dataset_name = "FourthBrainGenAI/MarketMail-AI"
product_name = "product"
product_desc = "description"
product_ad = "marketing_email"

dataset = load_dataset(dataset_name)
print(dataset)
print(dataset['train'][0])


##################################################################
# We want to put our data in the form:
# 
# ### INSTRUCTION
# Below is a product and description, please write a marketing email for this product.
#
# ### Product
# PRODUCT NAME
#
# ### Description:
# DESCRIPTION
#
# ### Marketing Email:
# OUR EMAIL HERE
#
#################################
# Say you were fine-tuning a Natural Language to SQL task.
# In that case - the data should be moved into the following format:
#
# ### INSTRUCTION
# Please convert the following context into an SQL query.
#
# ### CONTEXT
# How many people work at my company?
#
# ### SQL
# SELECT COUNT(*) FROM employees
##################################################################


### Because we're using BLOOMZ (which is an instruct-tuned base model), we should see better results providing the instruction - though that is not necessary.
def generate_prompt(product: str, description: str, marketing_email: str) -> str:
  prompt = f"### INSTRUCTION\nBelow is a product and description, please write a marketing email for this product.\n\n### Product:\n{product}\n### Description:\n{description}\n\n### Marketing Email:\n{marketing_email}"
  return prompt

mapped_dataset = dataset.map(lambda samples: tokenizer(generate_prompt(samples['product'], samples['description'], samples['marketing_email'])))


### If you're running into CUDA memory issues - please modify both the per_device_train_batch_size to be lower, and also reduce r in your LoRAConfig.
trainer = transformers.Trainer(
    model=model,
    train_dataset=mapped_dataset["train"],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=6,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        max_steps=100,
        learning_rate=1e-3,
        fp16=True,
        logging_steps=1,
        output_dir='outputs'
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()


### Share adapters on the 🤗 Hub
HUGGING_FACE_USER_NAME = "YOUR USERNAME HERE"

from huggingface_hub import notebook_login
notebook_login()

### If you run into issues during upload - please ensure you're using a HF Token with WRITE access!
model_name = "YOUR MODEL NAME HERE"

model.push_to_hub(f"{HUGGING_FACE_USER_NAME}/{model_name}", use_auth_token=True)


### Load adapters from the Hub
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

peft_model_id = f"{HUGGING_FACE_USER_NAME}/{model_name}"
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, return_dict=True, load_in_8bit=False)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_id)


### Inference
from IPython.display import display, Markdown

def make_inference(product, description):
  batch = tokenizer(f"### INSTRUCTION\nBelow is a product and description, please write a marketing email for this product.\n\n### Product:\n{product}\n### Description:\n{description}\n\n### Marketing Email:\n", return_tensors='pt')

  with torch.cuda.amp.autocast():
    output_tokens = model.generate(**batch, max_new_tokens=200)

  display(Markdown((tokenizer.decode(output_tokens[0], skip_special_tokens=True))))

your_product_name_here = "The Coolinator"
your_product_description_here = "A personal cooling device to keep you from getting overheated on a hot summer's day!"

make_inference(your_product_name_here, your_product_description_here)

