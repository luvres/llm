# Infrastructure Design for AI Consulting

### Build container image
```
apptainer build llm.sif llm.def
```
### Move image fot common path cluster
```
sudo mv llm.sif /scratch/images/
```

### Run (2.0.1+cu118 | True | Python 3.10.12 | CUDA 11.8.0)
```
export APPTAINER_BIND="/scratch/LLM/BLOOM/bloomz-3b/"
export IMAGE="/scratch/images/llm.sif"

apptainer run --nv ${IMAGE} python --version

apptainer run --nv ${IMAGE} python -c "import torch;print(torch.__version__)"

apptainer run --nv ${IMAGE} python -c "import torch;print(torch.cuda.is_available())"

apptainer shell --nv ${IMAGE}
```

## Download [Bloomz](https://huggingface.co/bigscience/bloomz)

#### bigscience/bloomz-3b (3B)
```
git lfs install
git clone https://huggingface.co/bigscience/bloomz-3b
```
#### bigscience/bloomz-7b1 (7.1B)
```
git lfs install
git clone https://huggingface.co/bigscience/bloomz-7b1
```
#### bigscience/bloomz (176B)
```
git lfs install
git clone https://huggingface.co/bigscience/bloomz
```

## Download [OpenLLaMA](https://github.com/openlm-research/open_llama)

#### openlm-research/open_llama_7b_v2
```
git lfs install
git clone https://huggingface.co/openlm-research/open_llama_7b_v2
```



