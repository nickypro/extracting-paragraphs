# %%
import json
from datetime import datetime
from os.path import exists
from typing import List, Tuple

import pandas as pd
import torch
from taker import Model

# set torch to use inference mode globally
torch.set_grad_enabled(False)

# %%
# m = Model("google/gemma-2-9b-it", dtype="bfp16")
# m = Model("meta-llama/Llama-3.2-3B-Instruct", dtype="bfp16")
# m = Model("google/gemma-2-2b-it", dtype="bfp16")
m = Model("meta-llama/Llama-3.1-8B-Instruct", dtype="bfp16")
m.show_details()

# %%
folder = f"../data/llama8b"
prefix = "V2"
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
max_new_tokens = 100
temperature = 0.3
neutral_prompt = "\n\n"
batch_size = 10

# %%

# DEFINE CODE FOR BATCHED GENERATION
from transformers import AutoTokenizer
m.tokenizer = AutoTokenizer.from_pretrained(m.tokenizer_repo, legacy=False, padding_side='left')
if m.tokenizer.pad_token is not None:
    pass
elif m.tokenizer.eos_token is not None:
    m.tokenizer.pad_token = m.tokenizer.eos_token
else:
    raise ValueError("Tokenizer has neither pad_token nor eos_token defined.")

def generate_batch_fast(m, batch_prompts, max_new_tokens, temperature) -> Tuple[List[str], List[str]]:
    # Tokenize all prompts in the batch
    batch_encodings = m.tokenizer(
        batch_prompts,
        padding=True,
        truncation=False,
        max_length=1000,
        return_tensors="pt"
    )
    orig_len = batch_encodings.input_ids.shape[1]
    # Generate outputs
    generate_ids = m.predictor.generate(
        input_ids=batch_encodings.input_ids.to(m.device),
        attention_mask=batch_encodings.attention_mask.to(m.device),
        max_length=batch_encodings.input_ids.shape[1] + max_new_tokens,
        do_sample=True,
        temperature=temperature,
        pad_token_id=m.tokenizer.pad_token_id,
    )
    # Decode all generated sequences at once
    batch_text_after = m.tokenizer.batch_decode(
        [ids[orig_len:] for ids in generate_ids],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    return batch_prompts, batch_text_after

def generate_batch_slow(m, batch_prompts, max_new_tokens, temperature) -> Tuple[List[str], List[str]]:
    text_ins = []
    text_outs = []

    for prompt in batch_prompts:
        print({"prompt": prompt})
        text_in, text_out = m.generate(prompt, max_new_tokens, temperature=temperature)
        text_ins.append(text_in)
        text_outs.append(text_out)

    return text_ins, text_outs

generate_batch = generate_batch_fast

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

[h.reset() for h in m.hooks.neuron_replace.values()] # RESET HOOKS BEFORE TRANSPLANTING NEXT SET OF ACTIVATIONS
for i in range(0, len(orig_df['part1']), batch_size):
    batch_prompts = orig_df['part1'][i:i+batch_size].tolist()  # Convert to list of strings
    batch_prompts, batch_outputs = generate_batch(m, batch_prompts, max_new_tokens, temperature)

    for prompt, output in zip(batch_prompts, batch_outputs):
        print({"output": output})
        data = {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "model": m.model_repo,
            "type": "original",
            "transplant_layers": None,
            "prompt": prompt,
            "output": output,
        }

        with open(filename, "a") as file:
            file.write(json.dumps(data) + "\n")

# %%
# TRANSFERRED GENERATIONS

transfer_scale = 1
transfer_tokens = 1
neutral_prompt = "\n\n"

filename = f"{folder}/{prefix}_transferred_{transfer_tokens}t_{transfer_scale}x_generation.jsonl"
if not exists(filename):
    with open(filename, "w") as f:
        pass

for info_prompt in orig_df['part1']:
    #RESET HOOKS BEFORE TRANSPLANTING NEXT SET OF ACTIVATIONS
    [h.reset() for h in m.hooks.neuron_replace.values()]
    acts = m.get_midlayer_activations(info_prompt)
    orig_token_index = m.get_ids(info_prompt).shape[1] - 1
    new_token_index  = m.get_ids(neutral_prompt).shape[1] - 1

    for layer_index in range(0,m.cfg.n_layers):
        m.hooks.neuron_replace[f"layer_{layer_index}_mlp_pre_out"].add_token(new_token_index, acts["mlp"][0, layer_index, orig_token_index]*transfer_scale)
        m.hooks.neuron_replace[f"layer_{layer_index}_attn_pre_out"].add_token(new_token_index, acts["attn"][0, layer_index, orig_token_index]*transfer_scale)

    output = m.generate(neutral_prompt, max_new_tokens, temperature=temperature)

    data = {
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "model": m.model_repo,
        "type": "transferred",
        "num_transferred_tokens": transfer_tokens,
        "transfer_scale": transfer_scale,
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

neutral_prompt = "\n\n"
def get_neutral_prompt(neutral, text, num_tokens):
    neutral_only_tokens = m.tokenizer.tokenize(neutral)
    neutral_cheat_tokens = m.tokenizer.tokenize(neutral+text)[:len(neutral_only_tokens)+num_tokens]
    neutral_cheat_ids = m.tokenizer.convert_tokens_to_ids(neutral_cheat_tokens)
    final_prompt = m.tokenizer.decode(neutral_cheat_ids)
    cheat_string = final_prompt[len(neutral):]
    return final_prompt, cheat_string

for neutralnum in [0]: #[1, 2, 5, 10, 15, 20]:
    filename = f"{folder}/{prefix}_neutral{neutralnum}_generation.jsonl"
    if not exists(filename):
        with open(filename, "w") as f:
            pass

    for i in range(0, len(orig_df['part2']), batch_size):
        batch = orig_df['part2'][i:i+batch_size]

        neutral_prompts = []
        cheat_strings = []
        for orig_output in batch:
            neutral_n, cheat_string = get_neutral_prompt(neutral_prompt, orig_output, neutralnum)
            neutral_prompts.append(neutral_n)
            cheat_strings.append(cheat_string)

        batch_prompts, batch_outputs = generate_batch(m, neutral_prompts, max_new_tokens, temperature)

        for j, (neutral_n, cheat_string, output) in enumerate(zip(neutral_prompts, cheat_strings, batch_outputs)):
            data = {
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "model": m.model_repo,
                "type": "neutral",
                "cheat_tokens": neutralnum,
                "transplant_layers": None,
                "cheat_prompt": neutral_n,
                "prompt": neutral_prompt,
                "output": cheat_string + output,
                "generated_output": output,
                "cheat_string": cheat_string,
            }

            with open(filename, "a") as file:
                file.write(json.dumps(data) + "\n")

# %%

# TRANSFERRED "IT'S" ACTIVATIONS PROMPT

transfer_scale = 1
transfer_tokens = 3

neutral_prompt = "\n\nIt's"

filename = f"{folder}/{prefix}_transferred_{transfer_tokens}t_{transfer_scale}x_generation.jsonl"
if not exists(filename):
    with open(filename, "w") as f:
        pass

for info_prompt in orig_df['part1']:
    # RESET HOOKS BEFORE TRANSPLANTING NEXT SET OF ACTIVATIONS
    [h.reset() for h in m.hooks.neuron_replace.values()]
    info_prompt = info_prompt + "It's"
    acts = m.get_midlayer_activations(info_prompt)
    orig_token_index = m.get_ids(info_prompt).shape[1] - 1
    new_token_index  = m.get_ids(neutral_prompt).shape[1] - 1

    for layer_index in range(0, m.cfg.n_layers):
        for t in range(transfer_tokens):
            m.hooks.neuron_replace[f"layer_{layer_index}_mlp_pre_out"].add_token(
                new_token_index-t,
                acts["mlp"][0, layer_index, orig_token_index-t] * transfer_scale
            )
            m.hooks.neuron_replace[f"layer_{layer_index}_attn_pre_out"].add_token(
                new_token_index-t,
                acts["attn"][0, layer_index, orig_token_index-t] * transfer_scale
            )

    prompt, text_output = m.generate(neutral_prompt, max_new_tokens, temperature=temperature)
    text_output = "It's" + text_output

    data = {
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "model": m.model_repo,
        "type": "transferred",
        "num_transferred_tokens": transfer_tokens,
        "transfer_scale": transfer_scale,
        "transplant_layers": (0, m.cfg.n_layers - 1),
        "orig_prompt": info_prompt,
        "transplant_prompt": neutral_prompt,
        "output": text_output,
    }

    with open(filename, "a") as file:
        file.write(json.dumps(data) + "\n")
# %%
