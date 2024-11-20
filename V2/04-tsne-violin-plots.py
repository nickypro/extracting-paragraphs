# %%
# Standard library imports
import json
from datetime import datetime
from os import listdir
from os.path import exists

# Third-party imports
import circuitsvis as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import phate
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
import torch
import umap
from tqdm import tqdm
from plotly.subplots import make_subplots
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import StandardScaler
from taker import Model
from taker.hooks import HookConfig
sns.set_theme()
torch.set_grad_enabled(False)

# Load sentence transformer
sentenceTransformer = SentenceTransformer("nvidia/NV-Embed-v2", trust_remote_code=True, model_kwargs={"load_in_8bit": True})
# sentenceTransformer = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# %%
# Load data and embeddings
prefix = "gemma9b/V2"
orig_df = pd.read_json(f"../data/{prefix}_orig_generation.jsonl", lines=True)
orig_df["output"] = orig_df["formatted_full_text"].apply(lambda text: text.split("\n\n")[1])

filepaths = {
    "Full Context":    f"../data/{prefix}_in-context_generation.jsonl",
    # "Transferred2": f"../data/{prefix}_transferred_2t_1x_generation.jsonl",
    # "Transferred2-100": f"../data/{prefix}_transferred_2t_100x_generation.jsonl",
    "Transferred": f"../data/{prefix}_transferred_1t_1x_generation.jsonl",
    # "Transferred1-100": f"../data/{prefix}_transferred_1t_100x_generation.jsonl",
    "Neutral0":    f"../data/{prefix}_neutral0_generation.jsonl",
    "Neutral1":    f"../data/{prefix}_neutral1_generation.jsonl",
    "Neutral2":    f"../data/{prefix}_neutral2_generation.jsonl",
    "Neutral5":    f"../data/{prefix}_neutral5_generation.jsonl",
    "Neutral10":   f"../data/{prefix}_neutral10_generation.jsonl",
    # "Neutral15":   f"../data/{prefix}_neutral15_generation.jsonl",
    # "Neutral20":   f"../data/{prefix}_neutral20_generation.jsonl",
}
dfs = {key: pd.read_json(filepath, lines=True) for key, filepath in filepaths.items()}
dfs["Neutral0"] = dfs["Neutral0"][:200]

def embed_outputs(outputs):
    embeddings = []
    batch_size = 5
    for i in tqdm(range(0, len(outputs), batch_size)):
        batch = outputs[i:i+batch_size]
        batch_embeddings = sentenceTransformer.encode(batch)
        embeddings.extend(batch_embeddings)
    return embeddings

orig_df["embeddings"] = embed_outputs(orig_df["output"].to_list())
for key, df in dfs.items():
    shorten = lambda text: text[:100] + text[100:].split("\n\n")[0]
    outputs = df["output"].apply(shorten)
    df["embeddings"] = embed_outputs(outputs.to_list())
    print(key, df["embeddings"].shape)

# %%
# VIOLIN PLOT OF ALL COSINE DISTANCES.
# COMPARE EACH df["embeddings"] TO orig_df["embeddings"].

# GET ALL COSINE DISTANCES
cosine_distances_list = []
comparison_labels = []
for key, df in dfs.items():
    cos = torch.nn.CosineSimilarity(dim=1)
    cossim = cos(torch.tensor(orig_df["embeddings"].tolist()), torch.tensor(df['embeddings'].tolist()))
    cosdist = 1 - cossim
    cosine_distances_list.append(cosdist.tolist())
    comparison_labels.append(key.capitalize())

violin_data = pd.DataFrame({
    'Cosine Distance': [dist for sublist in cosine_distances_list for dist in sublist],
    'Type of Generation': [comparison_labels[i] for i, sublist in enumerate(cosine_distances_list) for _ in sublist]
})

# MAKE VIOLIN PLOT
plt.figure(figsize=(9, 4))
sns.violinplot(data=violin_data,
               x='Type of Generation', y='Cosine Distance',
               split=False, hue="Type of Generation")
plt.ylabel("Cosine Distance")
plt.tight_layout()
sns.set_theme(font_scale=1.2)
plt.savefig('cosine_distances_violin_plot.png', dpi=600, bbox_inches='tight')
plt.ylim(0.0, None)

plt.show()

# %%
## COSINE DISTANCE METRICS
# Calculate mean and std of cosine distances for each type
metrics_df = pd.DataFrame({
    'Type': comparison_labels,
    'Mean': [np.mean(distances) for distances in cosine_distances_list],
    'Std': [np.std(distances) for distances in cosine_distances_list]
})

# Print formatted table
print("\nCosine Distance Metrics:")
print("-" * 80)
print(f"{'Metric':<10}", end='')
for type_name in metrics_df['Type']:
    print(f"{type_name:>10}", end='')
print()
print("-" * 80)
print(f"{'Mean':<10}", end='')
for mean in metrics_df['Mean']:
    print(f"{mean:>10.3f}", end='')
print()
print(f"{'Std':<10}", end='')
for std in metrics_df['Std']:
    print(f"{std:>10.3f}", end='')
print()
print("-" * 80)

# Print LaTeX table
print("\nLaTeX Table Format:")
print("\\begin{table}[h]")
print("\\centering")
print("\\begin{tabular}{l" + "c" * len(metrics_df) + "}")
print("\\toprule")
print("Metric & " + " & ".join(metrics_df['Type']) + " \\\\")
print("\\midrule")
print(f"Mean     & {' & '.join([f'{mean:.3f}' for mean in metrics_df['Mean']])} \\\\")
print(f"St. Dev. & {' & '.join([f'{std:.3f}' for std in metrics_df['Std']])} \\\\")
print("\\bottomrule")
print("\\end{tabular}")
print("\\caption{Cosine Distance Metrics}")
print("\\label{tab:cosine-metrics}")
print("\\end{table}")

# Print vertical LaTeX table
print("\nVertical LaTeX Table Format:")
print("\\begin{table}[h]")
print("\\centering")
print("\\begin{tabular}{lcc}")
print("\\toprule")
print("Type & Mean & St. Dev. \\\\")
for type_name, mean, std in zip(metrics_df['Type'], metrics_df['Mean'], metrics_df['Std']):
    print(f"{type_name} & {mean:.3f} & {std:.3f} \\\\")
print("\\bottomrule")
print("\\end{tabular}")
print("\\caption{Cosine Distance Metrics}")
print("\\label{tab:cosine-metrics-vertical}")
print("\\end{table}")


# %%

# TSNE PLOTS OF EACH df["embeddings"]

# Combine embeddings from all dataframes
filtered_dfs = {k: dfs[k] for k in ["Full Context", "Transferred", "Neutral0"]}
embeddings = np.vstack([df["embeddings"].tolist() for df in filtered_dfs.values()])
total_df = pd.concat([df.assign(**{'Type': key}) for key, df in filtered_dfs.items()], ignore_index=True)

# Perform t-SNE
reducer = TSNE(n_components=2)
tsne_embeddings_reduced = reducer.fit_transform(embeddings)
total_df["tsnedim1"] = tsne_embeddings_reduced[:, 0]
total_df["tsnedim2"] = tsne_embeddings_reduced[:, 1]

total_df["truncated_output"] = total_df["output"].str.slice(0, 25)
mini_df = total_df

# MAKE TSNE PLOT
########################
plt.figure(figsize=(6, 4))
fontsize = 15
scatter = sns.scatterplot(data=mini_df, x="tsnedim1", y="tsnedim2", hue="Type", alpha=0.8, palette="deep")
plt.xlabel("t-SNE dim1", fontsize=fontsize)
plt.ylabel("t-SNE dim2", fontsize=fontsize)
plt.gca().set_xticklabels([])
plt.gca().set_yticklabels([])
#plt.legend(bbox_to_anchor=(0.5, -0.20), loc='center', ncol=len(mini_df["Type"].unique()), framealpha=0, fontsize=fontsize*0.85)
plt.legend(bbox_to_anchor=(1.15, 0.5), loc='center left', ncol=1, framealpha=0, fontsize=fontsize*0.85)
# plt.title("t-SNE Plot of Embeddings")
plt.tight_layout()
plt.savefig("tsne-highres-all.png", dpi=600, bbox_inches='tight')
plt.show()

# %%
# Perform PHATE
########################
phate_op = phate.PHATE(n_components=2, verbose=False)
phate_embeddings_reduced = phate_op.fit_transform(embeddings)
total_df["phatedim1"] = phate_embeddings_reduced[:, 0]
total_df["phatedim2"] = phate_embeddings_reduced[:, 1]

total_df["truncated_output"] = total_df["output"].str.slice(0, 25)
mini_df = total_df
# mini_df = total_df[total_df["Type"].isin(["transferred", "original", "neutral0"])]
plt.figure(figsize=(5, 5))
fontsize = 15
scatter = sns.scatterplot(data=mini_df, x="phatedim1", y="phatedim2", hue="Type", alpha=0.8, palette="deep")
plt.xlabel("PHATE dim1", fontsize=fontsize)
plt.ylabel("PHATE dim2", fontsize=fontsize)
plt.gca().set_xticklabels([])
plt.gca().set_yticklabels([])
# plt.legend(bbox_to_anchor=(0.5, -0.20), loc='center', ncol=len(mini_df["Type"].unique()), framealpha=0, fontsize=fontsize*0.85)
# plt.legend(bbox_to_anchor=(1.0, 0.5), loc='center left', ncol=1, fontsize=fontsize*0.85)
plt.tight_layout()
plt.savefig("phate-highres-all.png", dpi=600, bbox_inches='tight')
plt.show()

# %%
# PRINT EXAMPLE OF PROMPT AND OUTPUTS
print(dfs["Full Context"]["prompt"][69][25:].strip())
for key, df in dfs.items():
    # outputs = []
    output = df["output"][69]
    print({key: output[:200] + output[200:].split("\n\n")[0]})

# %%
# PRINT EXAMPLE OF PROMPT AND OUTPUTS
prompt = dfs["Full Context"]["prompt"][69][25:].strip()
print("\\begin{tabular}{l|p{11cm}}")
print("\\toprule")
print("Context & Output \\\\")
print("\\midrule")
print(f"Prompt & {prompt} \\\\")
for key, df in dfs.items():
    output = df["output"][69]
    truncated = output[:200] + output[200:].split("\n\n")[0]
    print("\\midrule")
    print(f"{key} & {truncated} \\\\")
print("\\bottomrule")
print("\\end{tabular}")

# %%
