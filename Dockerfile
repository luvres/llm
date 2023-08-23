FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
LABEL mantainer="Leonardo Loures <luvres@hotmail.com>"

RUN \
  apt update && apt install --yes --no-install-recommends \
      python3-pip git && \
  \
  rm -rf /var/lib/apt/lists/* && \
  ln -s /usr/bin/python3 /usr/bin/python && \
  \
  pip3 --no-cache-dir install --upgrade pip && \
  \
  pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu118 && \
      torch torchvision torchaudio && \
      # --pre 'torch>=2.1.0dev'
  \
  pip install --no-cache-dir \
      git+https://github.com/huggingface/transformers.git \
      git+https://github.com/huggingface/accelerate.git \
      git+https://github.com/huggingface/peft.git \
      git+https://github.com/lvwerra/trl.git \
      bitsandbytes \
      datasets \
      loralib \
      scipy \
      openai
