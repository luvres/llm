import torch

print(f'CUDA:\n\
Avaliable: {torch.cuda.is_available()}\n\
 bfloat16: {torch.cuda.is_bf16_supported()}\n'\
)
