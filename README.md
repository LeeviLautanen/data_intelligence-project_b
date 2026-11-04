# Data Intelligence – Project B

- Model: [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B), loaded in 4-bit with --use_quantization

## Installation

Requires Python 3.11, an NVIDIA GPU with at least 8 GB of VRAM and about 20 GB free disk space.

```
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Test installation

Go into the PsySET folder and
```
cd ./PsySET
```

Test that the model loads and generates:
```
python main.py --model_index=13 --use_quantization --dataset=simple --steer_type=None
```

This should take under a minute and produce 10 answers.

## Creating steering vectors

Go into the PsySET folder and
```
cd ./PsySET
```

Create the steering vectors for anger with the command
```
python generate_steer_vecs.py --model_index=13 --use_quantization --dataset_source=goemotions_instruct_full_binary --target_tokens=all
```

The steering vectors should be generated in a few minutes. After the vectors are created, the program opens seven plots showing the probe accuracy per layer.