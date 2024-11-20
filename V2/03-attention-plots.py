# %%
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
from taker import Model
torch.set_grad_enabled(False)

pprint = lambda x: print(json.dumps(x, indent=4))

# %%
# LOAD MODEL and DATA
m = Model("google/gemma-2-9b-it", dtype="hqq8", device_map="cuda")
# m = Model("google/gemma-2-9b-it", dtype="bfp16")
m.show_details()

prefix = "gemma9b/V2"
orig_df = pd.read_json(f"../data/{prefix}_orig_generation.jsonl", lines=True)

# %%
# GET NEWLINE INDEX and TOKEN COUNT
for i, (prompt, text_out) in enumerate(zip(orig_df['formatted_input'], orig_df['output'])):
    text1 = text_out.split("\n\n")[0]
    text2 = "\n\n" + text_out.split("\n\n")[1]
    nn_index    = int(m.get_ids(prompt+text1).shape[-1])
    token_count = int(m.get_ids(prompt+text_out).shape[-1])
    orig_df.loc[i, "nn_index"] = nn_index
    orig_df.loc[i, "token_count"] = token_count

# %%
# GET EXAMPLE MEAN ATTENTION HEATMAP
attn = m.get_attn_weights(orig_df["formatted_full_text"][0], orig_df["nn_index"][0])

sns.heatmap(attn[:, :].detach().mean(dim=[0, 1, 2]).cpu().float().numpy(), center=0,vmax=0.02)
print(f"""newline index: {int(orig_df["nn_index"][0])} (out of {int(orig_df["token_count"][0])})""")
print(orig_df["formatted_full_text"][0].strip())
plt.axvline(orig_df["nn_index"][0], color="white", linestyle="--", alpha=0.5)
plt.axhline(orig_df["nn_index"][0], color="white", linestyle="--", alpha=0.5)
plt.title("Average Attention Pattern Across All Layers and Heads")
plt.xlabel("Key Position")
plt.ylabel("Query Position")
plt.show()

# %%
# GET MEAN ATTENTION HEATMAP
def plot_mean_attention_heatmap(m, df, window_size=50):
    plt.figure(figsize=(6, 6), dpi=600)
    plt.rcParams.update({'font.size': 18})

    all_mean_weights = []
    all_masks = []

    for _, row in df.iterrows():
        text = row['formatted_full_text']
        origin = int(row['nn_index'])

        attn = m.get_attn_weights(text.replace("\\n", "\n"))
        mean_weights = attn[:, :].detach().mean(dim=[0, 1, 2]).cpu().float().numpy()

        # Crop around origin
        start = int(max(0, origin - window_size))
        end = int(min(mean_weights.shape[0], origin + window_size))
        cropped = mean_weights[start:end, start:end]
        mask = np.ones_like(cropped)

        # Pad if needed
        pad_left = int(max(0, window_size - origin))
        pad_right = int(max(0, (origin + window_size) - mean_weights.shape[0]))
        padded_weights = np.pad(cropped, ((pad_left, pad_right), (pad_left, pad_right)), 'constant')
        padded_mask = np.pad(mask, ((pad_left, pad_right), (pad_left, pad_right)), 'constant')

        all_mean_weights.append(padded_weights)
        all_masks.append(padded_mask)

    # Average across examples
    stacked_weights = np.stack(all_mean_weights)
    stacked_masks = np.stack(all_masks)
    avg_weights = np.sum(stacked_weights * stacked_masks, axis=0) / (np.sum(stacked_masks, axis=0) + 1e-10)
    # Create heatmap with custom colorbar ticks
    hm = sns.heatmap(avg_weights, center=0.0, vmax=0.02, cmap="RdBu_r",
                     cbar_kws={'ticks': [0.0, 0.01, 0.02], 'format': '%.2f'})

    # Add center lines
    plt.axvline(x=window_size, color='white', alpha=0.5, linestyle=':')
    plt.axhline(y=window_size, color='white', alpha=0.5, linestyle=':')

    plt.xlabel("Token Position Relative to '\\n\\n'", fontsize=18)
    plt.ylabel("Token Position Relative to '\\n\\n'", fontsize=18)

    # Set ticks
    tick_locations = np.linspace(0, 2*window_size, 5)
    tick_labels = [f"{int(x - window_size):+d}" for x in tick_locations]
    plt.xticks(tick_locations, tick_labels, fontsize=18)
    plt.yticks(tick_locations, tick_labels, fontsize=18)

    plt.tight_layout()
    plt.show()

# Plot for original data
plot_mean_attention_heatmap(m, orig_df)


# %%
# GET ACTIVATION COSINE SIMILARITY HEATMAPS
def plot_layer_metrics_overlay_layers(m, df, layers_to_plot=None):
    window_size = 50  # Tokens to show on each side of origin
    if layers_to_plot is None:
        layers_to_plot = list(range(42))

    all_coss = [[] for _ in layers_to_plot]
    all_masks = [[] for _ in layers_to_plot]

    for _, row in df.iterrows():
        text = row['formatted_full_text']
        origin = int(row['nn_index'])

        with torch.no_grad():
            _, attn, _, _ = m.get_text_activations(text.replace("\\n", "\n"))

        for idx, layer in enumerate(layers_to_plot):
            vecs = attn[0][layer]
            norm = torch.norm(vecs, dim=-1, keepdim=True)
            norm = torch.where(norm == 0, torch.ones_like(norm), norm)
            normed_vecs = vecs / norm

            coss = torch.mm(normed_vecs, normed_vecs.t())

            start = max(0, origin - window_size)
            end = min(coss.shape[0], origin + window_size)
            cropped = coss[start:end, start:end]
            mask = torch.ones_like(cropped)

            pad_left = max(0, window_size - origin)
            pad_right = max(0, (origin + window_size) - coss.shape[0])
            padded_coss = torch.nn.functional.pad(cropped, (pad_left, pad_right, pad_left, pad_right))
            padded_mask = torch.nn.functional.pad(mask, (pad_left, pad_right, pad_left, pad_right))

            all_coss[idx].append(padded_coss)
            all_masks[idx].append(padded_mask)

    avg_coss = []
    for layer_coss, layer_masks in zip(all_coss, all_masks):
        stacked_coss = torch.stack(layer_coss)
        stacked_masks = torch.stack(layer_masks)
        avg = torch.sum(stacked_coss * stacked_masks, dim=0) / (torch.sum(stacked_masks, dim=0) + 1e-10)
        avg_coss.append(avg)

    num_plots = len(layers_to_plot)
    cols = min(6, num_plots)
    rows = (num_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3*rows), dpi=300)
    axes = np.array(axes).reshape(-1)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])

    fontsize = 12
    for idx, layer in enumerate(layers_to_plot):
        sns.heatmap(avg_coss[idx].cpu().float(), ax=axes[idx], square=True,
                   vmin=0, vmax=1, center=0, cbar=idx==0, cbar_ax=None if idx else cbar_ax)
        axes[idx].set_title(f"Layer {layer+1}", fontsize=fontsize*1.4)
        axes[idx].axvline(x=window_size, color='white', alpha=0.5, linestyle=':')
        axes[idx].axhline(y=window_size, color='white', alpha=0.5, linestyle=':')
        axes[idx].tick_params(labelbottom=False, labelleft=False)

    for i in range(num_plots, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout(rect=[0.03, 0.05, 0.9, 0.95])
    plt.show()

plot_layer_metrics_overlay_layers(m, orig_df, layers_to_plot=[0,5,11,17,23,29,35,41])

plot_layer_metrics_overlay_layers(m, orig_df)
# %%
