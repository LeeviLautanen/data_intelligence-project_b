import argparse
import json
import os

import torch
from LLMs.my_qwen3 import SteeredQwen3ForCausalLM
from my_datasets.emotion_eval import EmotionDataset
from steer_manager import (
    apply_wieghted_sum,
    get_default_steer_fn,
    get_steer_fn,
    load_steer_vectors,
)
from transformers import AutoTokenizer, BitsAndBytesConfig

parser = argparse.ArgumentParser()
parser.add_argument(
    "--name", required=True, help="Condition name, used as the output file name"
)
parser.add_argument("--steer_layers", type=str, default=None)
parser.add_argument(
    "--steer_coeff",
    type=float,
    default=5.0,
    required=False,
    help="Coefficient for intervention",
)
parser.add_argument(
    "--bs", type=int, default=16, required=False, help="Batch size for inference"
)

args = parser.parse_args()

MODEL_NAME = "Qwen/Qwen3-4B"
MODEL_SHORT_NAME = "Qwen3_4B_quantized"
CONCEPT = "anger"
CONCEPT_SOURCE = "goemotions_instruct_full_binary"
SYSTEM_PROMPT = "Pretend that you are a human."
MAX_NEW_TOKENS = 256  # Gotten from EmotionDataset.get_max_len
OUT_DIR = "../results/generations"

torch.manual_seed(1)

# Use PsySET's self report and ambiguous situation tasks as prompts
tasks = ["Self_report", "Ambiguous_situation_completion"]
data = [d for d in EmotionDataset().data if d["task"] in tasks]
print("Prompts:", len(data))

# Same model as main.py with --model_index=13 --use_quantization
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
quant = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = SteeredQwen3ForCausalLM.from_pretrained(
    MODEL_NAME, device_map="cuda", quantization_config=quant
)
model.eval()
model.generation_config.pad_token_id = tokenizer.pad_token_id

# Same steering as the defaults in main.py with --steer_type=Intervention --intervention_source=probeall
if args.steer_layers is None:
    model.set_steer_fn(get_default_steer_fn())
else:
    # We dont need all args from main.py, so use the arg defaults
    steer_vecs, steer_layers, steer_locs, steer_tokens = load_steer_vectors(
        MODEL_SHORT_NAME,
        model.config.num_hidden_layers,
        model.config.hidden_size,
        [CONCEPT],
        [CONCEPT_SOURCE],
        args.steer_layers,
        "7",
        "probeall",
        True,
    )
    steer_vec = apply_wieghted_sum(
        steer_vecs, torch.tensor([args.steer_coeff]), model.device
    )
    steer_fn = get_steer_fn(
        steer_vec,
        steer_layers,
        steer_locs,
        steer_tokens,
        model.config.hidden_size,
        "add",
        False,
    )
    model.set_steer_fn(steer_fn)

# Generate results in batches
results = []
for i in range(0, len(data), args.bs):
    # Form chat prompts for this batch
    batch = data[i : i + args.bs]
    prompts = []
    for d in batch:
        chat = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {"role": "user", "content": d["prompt"]},
        ]
        prompt = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

        prompts.append(prompt)

    # Generate responses
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    generation = model.generate(
        **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, disable_compile=True
    )

    # Decode response and add to results
    for d, seq in zip(batch, generation):
        response = tokenizer.decode(
            seq[inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        results.append({"task": d["task"], "prompt": d["prompt"], "response": response})
    print(f"{len(results)}/{len(data)}")

# Save the results
os.makedirs(OUT_DIR, exist_ok=True)
config = {
    "model": MODEL_NAME,
    "quantization": "4-bit",
    "concept": CONCEPT,
    "steer_layers": args.steer_layers,
    "steer_coeff": args.steer_coeff if args.steer_layers else None,
    "max_new_tokens": MAX_NEW_TOKENS,
    "system_prompt": SYSTEM_PROMPT,
}
out_path = f"{OUT_DIR}/{args.name}.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump({"config": config, "results": results}, f, indent=2, ensure_ascii=False)
print("Saved", out_path)
