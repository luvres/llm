Fine-Tuning LLM: QLoRA, bitsandbytes
-----
### Apptainer Build
```
apptainer build llm.sif llm.def
```
### Move image fot common path cluster
```
sudo mv llm.sif /scratch/images/
```

### Run (2.0.1+cu118 | True | Python 3.10.12 | CUDA 11.8.0)
```
export IMAGE="/scratch/images/llm.sif"  

apptainer run --nv ${IMAGE} python --version

apptainer run --nv ${IMAGE} python -c "import torch;print(torch.__version__)"

apptainer run --nv ${IMAGE} python -c "import torch;print(torch.cuda.is_available())"

apptainer shell --nv ${IMAGE}
```
