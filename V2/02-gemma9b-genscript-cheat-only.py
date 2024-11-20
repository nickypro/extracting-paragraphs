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

neutral_prompts = ["\n\n"]
neutral = neutral_prompts[0]
for neutralnum in [5, 10, 15, 20]:
    filename = f"../data/{prefix}_neutral{neutralnum}_generation.jsonl"
    if not exists(filename):
        with open(filename, "w") as f:
            pass

    for orig_output in orig_df['part2']:
        neutral_n, cheat_string = get_neutral_prompt(neutral, orig_output, neutralnum)
        output = m.generate(neutral_n, max_new_tokens, temperature=temperature)

        data = {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "model": "google/gemma-2-9b-it",
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
