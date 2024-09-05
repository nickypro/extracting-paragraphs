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
m.show_details()

# %%
with open('../data/promptsV1.csv', newline='') as f:
    reader = csv.reader(f)
    readdata = list(reader)
    readdata = readdata[:20]

import sys
import os
# %%
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# %%
# ORIGINAL GENERATIONS
filename = f"../datadata/latest_orig_generation.jsonl"
if not exists(filename):
    with open(filename, "w") as f:
        pass

max_new_tokens = 200
temperature = 0.3

[h.reset() for h in m.hooks.neuron_replace.values()] #RESET HOOKS BEFORE TRANSPLANTING NEXT SET OF ACTIVATIONS
for prompt in readdata:
    prompt = prompt[0]
    for i in range(10):
        output = m.generate(prompt, max_new_tokens, temperature=temperature)
        print(output)
        data = {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "model": "google/gemma-2-9b-it",
            "type": "original",
            "transplant_layers": None,
            "prompt": prompt,
            "output": output[1],
        }

        with open(filename, "a") as file:
            file.write(json.dumps(data) + "\n")

# NEUTRAL GENERATIONS
filename = f"../data/latest_neutral0_generation.jsonl"
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
            "model": "google/gemma-2-9b-it",
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
orig_df = pd.read_json(f"../data/latest_orig_generation.jsonl", lines=True)
def split_at_double_newline(text):
    # Ensure we are only working with strings longer than 15 characters
    if len(text) > 15:
        # Search for the first double newline after the 15th character
        pos = text.find('\n\n', 15)
        if pos != -1:  # Check if double newline was found
            return text[:pos+2], text[pos:]  # Split and remove the newline from the second part
    return text, None  # If no split is required, return the original text and None

# Apply the function to the DataFrame column
orig_df['paragraph1'], orig_df['paragraph2'] = zip(*orig_df['output'].apply(split_at_double_newline))
orig_df['paragraph1'] = orig_df['prompt'].astype(str) + orig_df['paragraph1'].astype(str)
print(repr(orig_df['paragraph1'][0]))
filename = f"../data/latest_transferred_generation.jsonl"
if not exists(filename):
    with open(filename, "w") as f:
        pass

for info_prompt in orig_df['paragraph1']:
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
            "model": "google/gemma-2-9b-it",
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

orig_df = pd.read_json(f"../data/latest_orig_generation.jsonl", lines=True)
def split_at_double_newline(text):
    # Ensure we are only working with strings longer than 15 characters
    if len(text) > 15:
        # Search for the first double newline after the 15th character
        pos = text.find('\n\n', 15)
        if pos != -1:  # Check if double newline was found
            return text[:pos+2], text[pos:]  # Split and remove the newline from the second part
    return text, text  # If no split is required, return the original text and None

orig_df['paragraph1'], orig_df['paragraph2'] = zip(*orig_df['output'].apply(split_at_double_newline))
orig_df['paragraph1'] = orig_df['prompt'].astype(str) + orig_df['paragraph1'].astype(str)

def get_neutral_prompt(text, num_tokens):
    idlist = m.get_ids(text).squeeze().tolist()
    neutral_tokens = m.tokenizer.convert_ids_to_tokens(idlist)
    neutral_tokens = [entry.replace("▁", " ") for entry in neutral_tokens]
    sep = ''
    return sep.join(neutral_tokens[1:num_tokens])


for neutralnum in [1, 2]:
    filename = f"../data/latest_neutral{neutralnum}_generation.jsonl"
    if not exists(filename):
        with open(filename, "w") as f:
            pass

    for orig_output in orig_df['paragraph2']:
        neutral_n = get_neutral_prompt(orig_output, neutralnum)
        output = m.generate(neutral_n, max_new_tokens, temperature=temperature)

        data = {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "model": "google/gemma-2-9b-it",
            "type": "neutral",
            "cheat_tokens": neutralnum,
            "transplant_layers": None,
            "prompt": neutral,
            "output": output[1],
        }

        with open(filename, "a") as file:
            file.write(json.dumps(data) + "\n")
