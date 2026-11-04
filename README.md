# Data Intelligence – Project B

- Model: [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B), loaded in 4-bit with --use_quantization

## Installation

Requires Python 3.11, an NVIDIA GPU and about 20 GB free disk space.

```
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Running

All commands are run from the PsySET folder with the venv activated.

Test that the model loads and generates:

```
python main.py --model_index=13 --use_quantization --dataset=simple --steer_type=None
```

This should take under a minute and produce 10 answers.
