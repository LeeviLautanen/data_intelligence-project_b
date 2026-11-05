# Data Intelligence – Project B

This is an extension to the PsySET repository. The extension generates steer vectors for anger and injects them into approximately the 25%, 50% and 75% points in the model's layers and then evaluates the results for anger, toxicity and quality. The model used is Qwen3-4B.

## Installation

Requires Python 3.11, an NVIDIA GPU with at least 8 GB of VRAM and about 20 GB free disk space.

```
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Run the generation and evaluation script

While in the repository root folder, run this script:
```
python run_all.py
```

This script will first generate the steering vectors which will take a few minutes. After the vectors have been generated, it will generate the responses using the vector injection approach used in this assignment into the folder `<repository-root>/results/generations`. Finally it will evaluate the generated responses based on the files in `<repository-root>/results/generations` and generate a comprehensive results csv and a summary csv into the folder `<repository-root>/results`.

The entire process is fully automated and takes approximately 15 minutes on an RTX 3080 10 GB.

## Artifact explanations in the results folder

generations/
- The responses from the model

results.csv
- Full results with prompt, response and evaluation

summary.csv
- Condensed summary for each condition

anger_rate.png
- Visualization of the ratio where anger was the dominant classified feeling

coherence.png
- Visualization of the coherence of responses
