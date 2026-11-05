import glob
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
from my_datasets.gpt_evals import GPTEvaluator
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

IN_DIR = "../results/generations"
OUT_DIR = "../results"
EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"
TOXICITY_MODEL = "s-nlp/roberta_toxicity_classifier"
JUDGE_MODEL = "Qwen/Qwen3-4B"
JUDGE_COLUMNS = ["coherence", "fluency", "engagingness", "refusal"]


# PsySET's quality judge but answered by a qwen3-4B instead of gpt-oss-20b
class LocalJudge(GPTEvaluator):
    def __init__(self):
        # Replace the init function so it uses our model
        self.test_mode = False
        self.offline = True
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(
            JUDGE_MODEL, device_map="cuda", quantization_config=quant
        )
        self.model.eval()

    def offline_inference(self, messages, max_new_tokens=60):
        # Override the inference function so it works with our judge model
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = self.tokenizer(text, return_tensors="pt").to("cuda")
        # Limit output size to 60 tokens, its enough for the one line of json
        out = self.model.generate(**inputs, max_new_tokens=60, do_sample=False)
        return self.tokenizer.decode(
            out[0, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )


# Load all conditions
rows = []
for path in sorted(glob.glob(f"{IN_DIR}/*.json")):
    condition = os.path.splitext(os.path.basename(path))[0]
    with open(path, encoding="utf-8") as f:
        for r in json.load(f)["results"]:
            rows.append({"condition": condition, **r})
df = pd.DataFrame(rows)
responses = df["response"].tolist()

# Estimate steering effectiveness with an emotion label and probability of anger
emotion = pipeline("text-classification", model=EMOTION_MODEL, top_k=None, device=0)
emo = emotion(responses, batch_size=16, truncation=True)
df["emotion_label"] = [max(e, key=lambda x: x["score"])["label"] for e in emo]
df["p_anger"] = [next(x["score"] for x in e if x["label"] == "anger") for e in emo]

# Estimate toxicity of response
toxicity = pipeline("text-classification", model=TOXICITY_MODEL, top_k=None, device=0)
tox = toxicity(responses, batch_size=16, truncation=True)
df["p_toxic"] = [next(x["score"] for x in t if x["label"] == "toxic") for t in tox]

# Judge the quality of responses using PsySET's judge scores
judge = LocalJudge()
for col in JUDGE_COLUMNS:
    df[col] = float("nan")
for i, row in tqdm(df.iterrows(), total=len(df), desc="Judge"):
    try:
        _, scores = judge.evaluate_quality(row["prompt"], row["response"])
        for col in JUDGE_COLUMNS:
            df.at[i, col] = pd.to_numeric(scores.get(col), errors="coerce")
    except ValueError:
        # Evaluation answer might not match required format
        pass
print("Unparseable judge replies:", df["coherence"].isna().sum())

# Save results and summary as csv files
os.makedirs(OUT_DIR, exist_ok=True)
df.to_csv(f"{OUT_DIR}/results.csv", index=False, encoding="utf-8")
summary = (
    df.groupby("condition")
    .agg(
        n=("response", "size"),
        anger_rate=("emotion_label", lambda s: (s == "anger").mean()),
        p_anger=("p_anger", "mean"),
        p_toxic=("p_toxic", "mean"),
        **{col: (col, "mean") for col in JUDGE_COLUMNS},
    )
    .round(3)
)
summary.to_csv(f"{OUT_DIR}/summary.csv", encoding="utf-8")
print(summary.to_string())

# Plot anger_rate and coherence
order = [
    "baseline",
    "layers_18_19_c5",
    "layers_9_10_c15",
    "layers_18_19_c15",
    "layers_27_28_c15",
]
summary = summary.loc[order]

plt.figure()
plt.bar(summary.index, summary["anger_rate"])
plt.title("Anger rate")
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/anger_rate.png")
plt.close()

plt.figure()
plt.bar(summary.index, summary["coherence"])
plt.title("Coherence")
plt.ylim(1, 5)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/coherence.png")
plt.close()
