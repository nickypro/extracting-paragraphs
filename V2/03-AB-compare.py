# %%
import json
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import torch

# Initialize sentence transformer
model = SentenceTransformer("all-mpnet-base-v2")
prefix = "AB3.1"

# Get embeddings for prompts A and B
# Read reference prompts and outputs
with open(f'/workspace/SPAR/extracting-paragraphs-copy/data/gemma9b/{prefix}_reference.jsonl', 'r') as f:
    reference = json.load(f)
    output_a = reference["output_a"]
    output_b = reference["output_b"]
    prompt_a = reference["prompt_a"]
    prompt_b = reference["prompt_b"]

embed_a = model.encode(output_a)
embed_b = model.encode(output_b)
# ... existing code until after embed_a and embed_b ...

# Function to process a single file
def process_file(filepath):
    layers = []
    similarities_a = []
    similarities_b = []

    # Dictionary to store max similarities for each swap index
    max_sims_by_swap = {}

    with open(filepath, 'r') as f:
        for line in f:
            data = json.loads(line)
            output = data['output']
            output = output.split("\n\n")[0]
            swap_index = data['swap_index']

            embed_output = model.encode(output)

            cos = torch.nn.CosineSimilarity(dim=0)
            sim_a = cos(torch.tensor(embed_output), torch.tensor(embed_a)).item()
            sim_b = cos(torch.tensor(embed_output), torch.tensor(embed_b)).item()

            if swap_index not in max_sims_by_swap:
                max_sims_by_swap[swap_index] = {'sim_a': sim_a, 'sim_b': sim_b, 'sim_a_max_text': output, 'sim_b_max_text': output}
            else:
                if sim_a > max_sims_by_swap[swap_index]['sim_a']:
                    max_sims_by_swap[swap_index]['sim_a'] = sim_a
                    max_sims_by_swap[swap_index]['sim_a_max_text'] = output

                if sim_b > max_sims_by_swap[swap_index]['sim_b']:
                    max_sims_by_swap[swap_index]['sim_b'] = sim_b
                    max_sims_by_swap[swap_index]['sim_b_max_text'] = output


    # Convert dictionary to sorted lists
    sorted_indices = sorted(max_sims_by_swap.keys())
    layers = sorted_indices
    similarities_a = [max_sims_by_swap[i]['sim_a'] for i in sorted_indices]
    similarities_b = [max_sims_by_swap[i]['sim_b'] for i in sorted_indices]

    #return layers, similarities_a, similarities_b
    return max_sims_by_swap

# Process both files
ab_max_sims_by_swap = process_file(f'/workspace/SPAR/extracting-paragraphs-copy/data/gemma9b/{prefix}_a_to_b_scrubbed_transferred_generations.jsonl')
ba_max_sims_by_swap = process_file(f'/workspace/SPAR/extracting-paragraphs-copy/data/gemma9b/{prefix}_b_to_a_scrubbed_transferred_generations.jsonl')

# Create the plot
plt.figure(figsize=(12, 6))
layers = list(ab_max_sims_by_swap.keys())

# Plot A->B file
plt.plot(layers, [ab_max_sims_by_swap[i]['sim_a'] for i in layers], label='A→B: Similarity to A', color='blue', alpha=0.7)
plt.plot(layers, [ab_max_sims_by_swap[i]['sim_b'] for i in layers], label='A→B: Similarity to B', color='red', alpha=0.7)

# Plot B->A file with dashed lines
plt.plot(layers, [ba_max_sims_by_swap[i]['sim_a'] for i in layers], label='B→A: Similarity to A', color='blue', alpha=0.7, linestyle='--')
plt.plot(layers, [ba_max_sims_by_swap[i]['sim_b'] for i in layers], label='B→A: Similarity to B', color='red', alpha=0.7, linestyle='--')

plt.xlabel('Layer Index')
plt.ylabel('Cosine Similarity')
plt.title('Cosine Similarity Across Layers')
plt.legend()
plt.grid(True, alpha=0.3)

# Add a horizontal line at y=1 for reference
plt.axhline(y=1, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
# %%
layers = list(ab_max_sims_by_swap.keys())
for i in layers:
    print({"layer": i, "a_max_text": ab_max_sims_by_swap[i]['sim_a_max_text'], "b_max_text": ab_max_sims_by_swap[i]['sim_b_max_text']})
# %%
