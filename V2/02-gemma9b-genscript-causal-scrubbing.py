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
hook_config = """
pre_decoder: replace, collect
post_decoder: replace, collect
post_attn: replace, collect
"""
m = Model("google/gemma-2-9b-it", dtype="bfp16", hook_config=hook_config)
m.show_details()

# %%
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
prefix = "gemma9b/AB3.1"
neutral_prompt = "\n\n"
max_new_tokens = 100
temperature = 0.3
n_copies = 100
scale_factor = 1
n_tokens_to_transfer = 1

prompt_template = "<bos><start_of_turn>user\nTell me about cleaning your house in 50 words, then tell me about %s in 50 words. Structure it as 2 paragraphs, 1 paragraph each. Do NOT explicitly name either topic.<end_of_turn>\n<start_of_turn>assistant\nMaintaining a clean and organized home is essential for a healthy and peaceful living environment. Regularly tidying up, dusting, and vacuuming can make a significant difference. Creating routines and habits can help keep clutter at bay, allowing you to relax and focus on more enjoyable activities.\n\n"
prompt_a = prompt_template % "polycystic kidney disease"
prompt_b = prompt_template % "monster trucks"

# %%
# GET ORIGINAL GENERATIONS FOR REFERENCE
input_a, output_a = m.generate(prompt_a, max_new_tokens, temperature=temperature, max_new_tokens=max_new_tokens)
input_b, output_b = m.generate(prompt_b, max_new_tokens, temperature=temperature, max_new_tokens=max_new_tokens)

print({"a": output_a})
print({"b": output_b})

filename = f"../data/{prefix}_reference.jsonl"
if not exists(filename):
    with open(filename, "w") as f:
        pass

# save untampered outputs
with open(filename, "a") as file:
    file.write(json.dumps({"prompt_a": prompt_a, "prompt_b": prompt_b, "output_a": output_a, "output_b": output_b}) + "\n")


# %%
# TRANSFERRED GENERATIONS

for direction in ["a_to_b", "b_to_a"]:
    if direction == "a_to_b":
        prompt_a = prompt_template % "polycystic kidney disease"
        prompt_b = prompt_template % "monster trucks"
    else:
        prompt_a = prompt_template % "monster trucks"
        prompt_b = prompt_template % "polycystic kidney disease"

    filename = f"../data/{prefix}_{direction}_scrubbed_transferred_generations.jsonl"
    if not exists(filename):
        with open(filename, "w") as f:
            pass

    # GET ORIGINAL ACTIVATIONS FOR REFERENCE
    [h.reset() for h in m.hooks.neuron_replace.values()]
    m.hooks.disable_all_collect_hooks()
    m.hooks.enable_collect_hooks(["pre_decoder", "post_decoder", "post_attn"])
    act_data = {}
    for prompt_label, prompt in [("a", prompt_a), ("b", prompt_b)]:
        orig_token_index = m.get_ids(prompt).shape[1] - 1
        m.get_outputs_embeds(prompt)
        pre_resid_0     = m.collect_recent_pre_decoder()[:, 0]
        post_resid_acts = m.collect_recent_post_decoder()[:, :]
        post_attn_acts  = m.collect_recent_post_attn()
        act_data[prompt_label] = {
            "orig_token_index": orig_token_index,
            "pre_resid_0": pre_resid_0,
            "post_resid_acts": post_resid_acts,
            "post_attn_acts": post_attn_acts,
        }


    for a_to_b_index in range(m.cfg.n_layers+1):
        for _ in range(n_copies):
            #RESET HOOKS BEFORE TRANSPLANTING NEXT SET OF ACTIVATIONS
            [h.reset() for h in m.hooks.neuron_replace.values()]
            neutral_prompt = ""

            for i in range(n_tokens_to_transfer):
                # do layer zero
                prompt_label = "b" if a_to_b_index > 0 else "a"
                new_token_index  = m.get_ids(neutral_prompt).shape[1] - 1 - i
                orig_token_index = act_data[prompt_label]["orig_token_index"]
                m.hooks.neuron_replace[f"layer_0_post_decoder"].add_token(new_token_index, act_data[prompt_label]["pre_resid_0"][0, orig_token_index]*scale_factor)

                # now do the rest of the layers
                for layer_index in range(m.cfg.n_layers):
                    prompt_label = "b" if a_to_b_index > layer_index else "a"
                    orig_token_index = act_data[prompt_label]["orig_token_index"]
                    m.hooks.neuron_replace[f"layer_{layer_index}_post_decoder"].add_token(new_token_index, act_data[prompt_label]["post_resid_acts"][0, layer_index, orig_token_index]*scale_factor)
                    m.hooks.neuron_replace[f"layer_{layer_index}_post_attn"].add_token(new_token_index, act_data[prompt_label]["post_attn_acts"][0, layer_index, orig_token_index]*scale_factor)

                output = m.generate(neutral_prompt, max_new_tokens, temperature=temperature)

                data = {
                    "temperature": temperature,
                    "max_new_tokens": max_new_tokens,
                    "swap_index": a_to_b_index,
                    "model": "google/gemma-2-9b-it",
                    "type": "transferred",
                    "num_transferred_tokens": 1,
                    "transplant_layers": (0,m.cfg.n_layers-1),
                    "orig_prompt": neutral_prompt,
                    "transplant_prompt": neutral_prompt,
                    "output": output[1],
                }

                with open(filename, "a") as file:
                    file.write(json.dumps(data) + "\n")

    # %%
