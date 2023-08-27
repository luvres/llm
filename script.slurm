#!/bin/bash
#SBATCH --partition=gpu-rtx3090      # -p
#SBATCH --gres=gpu:rtx3090:2         # 1 GPU per node
#
##SBATCH --partition=gpu-titan-v     # -p
##SBATCH --gres=gpu:titanv:1         # 1 GPU per node
#
#SBATCH --constraint=GPU             # -C feature
#SBATCH --nodes=1                    # -N | node count, 1 nodes reserved
#SBATCH --time=7-00:00:00            # -t | total run time limit (d-HH:MM:SS)
#SBATCH --output=./logs/slurm.log    # Standard output and error log
#SBATCH --error=./logs/slurm.log
###########
export APPTAINER_BIND="/scratch/$USER,/scratch/LLM"
export CONTAINER_FILE="/opt/images/llm.sif"
###########
# 
# Options:
# --model_path, example: 'BLOOM' | 'OLLAMA'
# --model_name, example: 'bloomz-7b1' | 'open_llama_3b_v2'
# --peft_method, choices: {'lora','qlora'}
# --tuning, choices: {'adapter', 'instruction'}
# --lora_target_modules: ["query_key_value"] ----------> (bloom | gpt_neox | chatglm)
#                        ["q_proj", "v_proj"] ---------> (llama | bart | opt | gptj | gpt_neo)
#                        ["q", "v"] -------------------> (t5 | mt5 )
#                        ["catt_n"] -------------------> (gpt2)
#                        ["query", "value"] -----------> (bert | roberta | xlm-roberta | electra | layoutlm)
#                        ["query_proj", "value_proj"] -> (deberta-v2)
#                        ["in_proj"] ------------------> (deberta)
apptainer run --nv ${CONTAINER_FILE} python -u main.py \
--model_path 'OLLAMA' \
--model_name 'open_llama_3b_v2' \
--peft_method "qlora" \
\
--tuning 'instruction' \
\
--inference "Qual o sentido da vida?" \
--max_new_tokens 1000 \
\
--lora_r 16 \
--lora_alpha 32 \
--lora_target_modules '' \
--lora_dropout 0.05  \
--lora_bias 'none' \
--lora_task_type 'CAUSAL_LM' \
\
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 4 \
--max_steps 100 \
--learning_rate 1e-4

