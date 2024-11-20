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
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
prefix = "V2"
max_new_tokens = 100
temperature = 0.3
n_copies = 1
scale_factor = 1
n_tokens_to_transfer = 1
neutral_prompts = ["\n\n"]


# %%
# ORIGINAL GENERATIONS
# These seem pretty consistently formatted, so we can just split on the first
# double newline, which is after the first paragraph.
orig_df = pd.read_json(f"../data/{prefix}_orig_generation.jsonl", lines=True)
split_before_newline = lambda x: x.split('\n\n')[0] + "\n\n"
split_after_newline  = lambda x: x.split('\n\n')[1]
orig_df['part1'] = orig_df['formatted_full_text'].apply(split_before_newline)
orig_df['part2'] = orig_df['formatted_full_text'].apply(split_after_newline)
print({"start": orig_df['part1'][0]})
print({"end": orig_df['part2'][0]})

# %%
# TRANSFERRED GENERATIONS

filename = f"../data/{prefix}_transferred_{n_tokens_to_transfer}t_{scale_factor}x_generation.jsonl"
if not exists(filename):
    with open(filename, "w") as f:
        pass

for info_prompt in orig_df['part1']:
    #RESET HOOKS BEFORE TRANSPLANTING NEXT SET OF ACTIVATIONS
    [h.reset() for h in m.hooks.neuron_replace.values()]
    acts = m.get_midlayer_activations(info_prompt)
    for neutral_prompt in neutral_prompts:
        for i in range(n_tokens_to_transfer):
            orig_token_index = m.get_ids(info_prompt).shape[1] - 1 - i
            new_token_index  = m.get_ids(neutral_prompt).shape[1] - 1 - i
            for layer_index in range(0,m.cfg.n_layers):
                m.hooks.neuron_replace[f"layer_{layer_index}_mlp_pre_out"].add_token(new_token_index, acts["mlp"][0, layer_index, orig_token_index]*scale_factor)
                m.hooks.neuron_replace[f"layer_{layer_index}_attn_pre_out"].add_token(new_token_index, acts["attn"][0, layer_index, orig_token_index]*scale_factor)

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
