FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
LABEL mantainer="Leonardo Loures <luvres@hotmail.com>"

## References:
# https://blog.ovhcloud.com/fine-tuning-llama-2-models-using-a-single-gpu-qlora-and-ai-notebooks/
# https://github.com/artidoro/qlora/tree/main

RUN \
  apt-get update && apt-get install --yes --no-install-recommends git \
  \
  && pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir \
        git+https://github.com/huggingface/accelerate.git \
        git+https://github.com/huggingface/transformers.git \
        git+https://github.com/huggingface/peft.git \
        git+https://github.com/lvwerra/trl.git \
        bitsandbytes \
        scipy \
        datasets==2.13.1
