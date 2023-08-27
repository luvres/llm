import os
import argparse
import torch
import torch.nn as nn
#import bitsandbytes as bnb
from transformers import (
    AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, LlamaTokenizer,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset, Dataset, DatasetDict
from trl import SFTTrainer

#os.environ["CUDA_VISIBLE_DEVICES"]="0"
print(f'CUDA Avaliable: {torch.cuda.is_available()}')

# Environment variable
USER = os.environ['USER']

parser = argparse.ArgumentParser(description="Prepared Adapter")
parser.add_argument('--model_path', type=str, help="Name of the model, example: 'BLOOM'", required=True)
parser.add_argument('--model_name', type=str, help="Name of the model, example: 'bloomz-3b'", required=True)
parser.add_argument('--peft_method', type=str, choices={'lora','qlora'}, default='qlora')
parser.add_argument('--tuning', type=str, choices={'adapter','instruction'}, default='adapter')
parser.add_argument('--inference', type=str, default="Quais são as estações do ano?")
parser.add_argument('--max_new_tokens', type=int, default=50)
parser.add_argument('--lora_r', type=int, default=16)
parser.add_argument('--lora_alpha', type=int, default=32)
parser.add_argument('--lora_target_modules', type=str, default='query_key_value')
parser.add_argument('--lora_dropout', type=float, default=0.05)
parser.add_argument('--lora_bias', type=str, choices={'all','none'}, required='none')
parser.add_argument('--lora_task_type', type=str, default='CAUSAL_LM')
parser.add_argument('--per_device_train_batch_size', type=int, default=6)
parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
parser.add_argument('--warmup_steps', type=int, default=100)
parser.add_argument('--max_steps', type=int, default=100)
parser.add_argument('--learning_rate', type=float, default=1e-4) # 
args = parser.parse_args()

model_name = args.model_name
model_path = args.model_path
peft_method = args.peft_method
tuning = args.tuning
inference = args.inference
max_new_tokens = args.max_new_tokens
# LoraConfig
lora_r = args.lora_r
lora_alpha = args.lora_alpha
lora_target_modules = args.lora_target_modules
lora_dropout = args.lora_dropout
lora_bias = args.lora_bias
lora_task_type = args.lora_task_type
# Train adapter-tuning
per_device_train_batch_size = args.per_device_train_batch_size
gradient_accumulation_steps = args.gradient_accumulation_steps
warmup_steps = args.warmup_steps
max_steps = args.max_steps
learning_rate = args.learning_rate

model_id = f"/scratch/LLM/{model_path}/{model_name}"
model_pretrained =  f"/scratch/{USER}/adapters/{model_name}-{peft_method}"

# qLoRA
if peft_method == 'qlora':
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
# LoRA
elif peft_method == 'lora':
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        load_in_8bit=True,
        device_map='auto',
    )

qlora_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
#    target_modules=[lora_target_modules],
    lora_dropout=lora_dropout,
    bias=lora_bias,
    task_type=lora_task_type
)

# Tokenizer
if tuning == 'adapter':
    tokenizer = AutoTokenizer.from_pretrained(model_id)
# Supervised fine-tuning
elif tuning == 'instruction':
    tokenizer = LlamaTokenizer.from_pretrained(model_id)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# qLoRA
if peft_method == 'qlora' and tuning == 'adapter':
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, qlora_config)
# LoRA
elif peft_method == 'lora':
    # Freezing the original wheigths 
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

    model = get_peft_model(model, qlora_config)
    

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

# Load dataset
#DatasetDict({
#    train: Dataset({
#        features: ['user', 'chip2'],
#        num_rows: 210289
#    })
#})
dataset_name = "paulofinardi/OIG_small_chip2_portuguese_brasil"
dataset = load_dataset(dataset_name)

dataset_reduced = DatasetDict({
    "train": Dataset.from_dict({
        "user": dataset["train"]["user"][:1000],
        "chip2": dataset["train"]["chip2"][:1000]
    })
})

dataset_prepared = dataset_reduced["train"].train_test_split(test_size=0.1)

if tuning == 'adapter':
#def generate_prompt(user: str, chip2: str) -> str:
    def create_prompt(question, answer):
      if len(answer) < 1:
          chip2 = "Cannot Find Answer"
      else:
          answer = answer
      prompt_template = f"### QUESTION\n{question}\n\n### ANSWER\n{answer}</s>"
      return prompt_template

    mapped_dataset = dataset_reduced.map(lambda samples: tokenizer(create_prompt(samples['user'],     samples['chip2'])))
elif tuning == 'instruction':
    def create_prompt(example):
        if example.get("answer", "") != "":
            input_prompt = (f"Não é possível encontrar a resposta.")
        else:
            prompt_template = (
            "### QUESTION\n"
            f"{example['user']}\n\n"
            "### ANSWER\n"
            f"{example['chip2']}")
        return {"text" : prompt_template}
    mapped_dataset = dataset_prepared.map(create_prompt)

# Unsupervised fine-tuning
if tuning == 'adapter':
    trainer = Trainer(
        model=model,
        train_dataset=mapped_dataset["train"],
        args=TrainingArguments(
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=1,
            output_dir='outputs'
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()
    model.save_pretrained(model_pretrained)
# Supervised fine-tuning
elif tuning == 'instruction':
    training_arguments=TrainingArguments(
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_steps=max_steps,
        learning_rate=learning_rate,
        fp16=True,
        optim="paged_adamw_8bit",
        output_dir='outputs'
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=mapped_dataset["train"],
        eval_dataset=mapped_dataset["test"],
        tokenizer=tokenizer,
        peft_config=qlora_config,
        dataset_text_field="text",
        max_seq_length=512,
        args=training_arguments
    )
    trainer.train()
    trainer.save_model(model_pretrained)


# Inference
if tuning == 'adapter':
    def make_inference():
        batch = tokenizer(inference, return_tensors='pt')
        with torch.cuda.amp.autocast():
            output_tokens = model.generate(**batch, max_new_tokens=max_new_tokens)
        print('\n\n', tokenizer.decode(output_tokens[0], skip_special_tokens=True))
elif tuning == 'instruction':
    def make_inference(question):
        print(question)
#        prompt = f"### QUESTION\n{question}\n\n### ANSWER\n"
#        inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to("cuda:0")
#        outputs = model.generate(**inputs, max_new_tokens=1000)
#        print('\n\n', tokenizer.decode(outputs[0], skip_special_tokens=True))
##        print("---- NON-INSTRUCT-TUNED-MODEL ----")
##        print('\n\n', tokenizer.decode(outputs[0], skip_special_tokens=True))

make_inference(inference)

