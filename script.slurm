#!/bin/bash
##SBATCH --partition=gpu-rtx3090     # -p
##SBATCH --gres=gpu:rtx3090:2        # 1 GPU per node
#
#SBATCH --partition=gpu-titan-v      # -p
#SBATCH --gres=gpu:titanv:1          # 1 GPU per node
#
#SBATCH --constraint=GPU             # -C feature
#SBATCH --nodes=1                    # -N | node count, 1 nodes reserved
#SBATCH --time=2-00:00:00            # -t | total run time limit (d-HH:MM:SS)
#SBATCH --output=./logs/slurm.log    # Standard output and error log
#SBATCH --error=./logs/slurm.log
###########
export APPTAINER_BIND="/scratch/$USER,/scratch/LLM"
export CONTAINER_FILE="/opt/images/llm.sif"
###########
#apptainer run --nv ${CONTAINER_FILE} python -u peft-lora.py

apptainer run --nv ${CONTAINER_FILE} python -u adapter.py \
--model_name "bloomz-7b1" \
--peft_method "qlora"
