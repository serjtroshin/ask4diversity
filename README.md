This is supplementary code for the paper ["Asking a Language Model for Diverse Responses"](https://arxiv.org/abs/2509.17570) presented at the Second Workshop on Uncertainty-Aware NLP at EMNLP 2025.

# Installation

## 1. Create a virtual environment:
```bash
conda create -n sampling_workshop python=3.10.15              
conda activate sampling_workshop
```

## 2. Install dependencies via Poetry:
```bash
# 1. (Optional) Pick a specific Python interpreter for your venv:
conda create -n sampling_workshop python=3.10.15
conda activate sampling_workshop

# 2. Install all dependencies (and Poetry will create a venv if one doesn’t exist):
poetry install

# 3. Spawn a shell inside that venv:
poetry shell
```

## Usage

### Generation
```bash
# Default run
PYTHONPATH='.' python src/run_generation.py --config-name run

# With specific model and seed
PYTHONPATH='.' python src/run_generation.py --config-name run model=qwen3-1.7b generation.seed=1

# Hyperparameter sweeping
PYTHONPATH='.' python src/run_generation.py --config-name run -m generation.seed=1,2,3
```


### Diversity evaluation
```bash
PYTHONPATH='.' python src/run_diversity.py --config-name run seed=1
```

### Arithmetic diversity evaluation
```bash
PYTHONPATH='.' python src/run_arithmetic_diversity.py --config-name run seed=1
```

## Running Paper Experiments

Use the scripts in `run/` directory to reproduce the experiments from the paper. Choose between SLURM (for cluster environments) or bash mode (for local execution): 

```bash
# Generation experiments
bash run/all_gen.sh [model] [python_path] --bash

# Diversity evaluation
bash run/all_div.sh [model] [python_path] --bash

# Arithmetic diversity evaluation
bash run/all_arithmetic_div.sh [model] [python_path] --bash
```

**Parameters:**
- `model`: Model name (default: "qwen4b")
- `python_path`: Python interpreter path (default: "python")
- `--bash`: Use bash mode instead of SLURM (add this flag for local execution)


## Citation
If you find this code useful for your research, please consider citing our paper:
```
@inproceedings{
troshin2025asking,
title={Asking a Language Model for Diverse Responses},
author={Sergey Troshin and Irina Saparina and Antske Fokkens and Vlad Niculae},
booktitle={Second Workshop on Uncertainty-Aware NLP - EMNLP 2025},
year={2025},
url={https://openreview.net/forum?id=Tf73rORM0x}
}
```
