# %%
import csv
import json
from datetime import datetime
from os import listdir
from os.path import exists

import circuitsvis as cv
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from taker import Model
from taker.hooks import HookConfig

# %%
m = Model("google/gemma-2-9b-it", dtype="bfp16")
# m = Model("meta-llama/Llama-3.2-3B-Instruct", dtype="bfp16")
m.show_details()

# %%
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
folder = f"../data/gemma9b"
prefix = "V2.1"


# %%
# ORIGINAL GENERATIONS
# These seem pretty consistently formatted, so we can just split on the first
# double newline, which is after the first paragraph.
orig_df = pd.read_json(f"{folder}/{prefix}_orig_generation.jsonl", lines=True)
split_before_newline = lambda x: x.split('\n\n')[0] + "\n\n"
split_after_newline  = lambda x: x.split('\n\n')[1]
orig_df['part1'] = orig_df['formatted_full_text'].apply(split_before_newline)
orig_df['part2'] = orig_df['formatted_full_text'].apply(split_after_newline)
print({"start": orig_df['part1'][0]})
print({"end": orig_df['part2'][0]})

# %%

# COPY IN-CONTEXT GENERATIONS
filename = f"{folder}/{prefix}_in-context_generation.jsonl"
if not exists(filename):
    with open(filename, "w") as f:
        pass

max_new_tokens = 100
temperature = 0.3
n_copies = 1

[h.reset() for h in m.hooks.neuron_replace.values()] #RESET HOOKS BEFORE TRANSPLANTING NEXT SET OF ACTIVATIONS
for prompt in orig_df['part1']:
    for i in range(n_copies):
        output = m.generate(prompt, max_new_tokens, temperature=temperature)
        print(output)
        data = {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "model": m.model_repo,
            "type": "original",
            "transplant_layers": None,
            "prompt": prompt,
            "output": output[1],
        }

        with open(filename, "a") as file:
            file.write(json.dumps(data) + "\n")

# %%

# NEUTRAL GENERATIONS
filename = f"{folder}/{prefix}_neutral0_generation.jsonl"
if not exists(filename):
    with open(filename, "w") as f:
        pass

neutral_prompts = ["\n\n"]
for neutral in neutral_prompts:
    for i in range(1000):
        output = m.generate(neutral, max_new_tokens, temperature=temperature)

        data = {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "model": m.model_repo,
            "type": "neutral",
            "cheat_tokens": 0,
            "transplant_layers": None,
            "prompt": neutral,
            "output": output[1],
        }

        with open(filename, "a") as file:
            file.write(json.dumps(data) + "\n")


# %%
# TRANSFERRED GENERATIONS

filename = f"{folder}/{prefix}_transferred1_generation.jsonl"
if not exists(filename):
    with open(filename, "w") as f:
        pass

neutral_prompts = ["\n\n"]
for info_prompt in orig_df['part1']:
    #RESET HOOKS BEFORE TRANSPLANTING NEXT SET OF ACTIVATIONS
    [h.reset() for h in m.hooks.neuron_replace.values()]
    acts = m.get_midlayer_activations(info_prompt)
    orig_token_index = m.get_ids(info_prompt).shape[1] - 1
    for neutral_prompt in neutral_prompts:
        new_token_index  = m.get_ids(neutral_prompt).shape[1] - 1

        for layer_index in range(0,m.cfg.n_layers):
            m.hooks.neuron_replace[f"layer_{layer_index}_mlp_pre_out"].add_token(new_token_index, acts["mlp"][0, layer_index, orig_token_index]*100)
            m.hooks.neuron_replace[f"layer_{layer_index}_attn_pre_out"].add_token(new_token_index, acts["attn"][0, layer_index, orig_token_index]*100)

        output = m.generate(neutral_prompt, max_new_tokens, temperature=temperature)

        data = {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "model": m.model_repo,
            "type": "transferred",
            "num_transferred_tokens": 1,
            "transplant_layers": (0,m.cfg.n_layers-1),
            "orig_prompt": info_prompt,
            "transplant_prompt": neutral_prompt,
            "output": output[1],
        }

        with open(filename, "a") as file:
            file.write(json.dumps(data) + "\n")

# %%
# CHEATING NEUTRAL GENERATIONS

# Reset any possible hook values
[h.reset() for h in m.hooks.neuron_replace.values()]

def get_neutral_prompt(neutral, text, num_tokens):
    neutral_only_tokens = m.tokenizer.tokenize(neutral)
    neutral_cheat_tokens = m.tokenizer.tokenize(neutral+text)[:len(neutral_only_tokens)+num_tokens]
    neutral_cheat_ids = m.tokenizer.convert_tokens_to_ids(neutral_cheat_tokens)
    final_prompt = m.tokenizer.decode(neutral_cheat_ids)
    cheat_string = final_prompt[len(neutral):]
    return final_prompt, cheat_string

neutral = neutral_prompts[0]
for neutralnum in [1, 2, 5, 10]:
    filename = f"{folder}/{prefix}_neutral{neutralnum}_generation.jsonl"
    if not exists(filename):
        with open(filename, "w") as f:
            pass

    for orig_output in orig_df['part2']:
        neutral_n, cheat_string = get_neutral_prompt(neutral, orig_output, neutralnum)
        output = m.generate(neutral_n, max_new_tokens, temperature=temperature)

        data = {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "model": m.model_repo,
            "type": "neutral",
            "cheat_tokens": neutralnum,
            "transplant_layers": None,
            "cheat_prompt": neutral_n,
            "prompt": neutral,
            "output": cheat_string + output[1],
            "generated_output": output[1],
            "cheat_string": cheat_string,
        }

        with open(filename, "a") as file:
            file.write(json.dumps(data) + "\n")

# %%
