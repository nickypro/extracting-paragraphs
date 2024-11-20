# %%
# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import phate
import plotly.express as px
import plotly.io as pio
import seaborn as sns
import torch
from plotly.subplots import make_subplots
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import StandardScaler
from taker import Model

# Load sentence transformer
sentenceTransformer = SentenceTransformer("all-mpnet-base-v2")

# %%
# Load data and embeddings
# folder = "../data/llama8b"
# prefix = "V2"
# folder = "../data/llama3b"
# prefix = "V2.1"
folder = "../data/ministral3b"
prefix = "V2"
orig_df = pd.read_json(f"{folder}/{prefix}_orig_generation.jsonl", lines=True)
orig_df["output"] = orig_df["formatted_full_text"].apply(lambda text: text.split("\n\n")[1])

filepaths = {
    "original":    f"{folder}/{prefix}_in-context_generation.jsonl",
    "transferred1": f"{folder}/{prefix}_transferred_1t_1x_generation.jsonl",
    "transferred3": f"{folder}/{prefix}_transferred_3t_1x_generation.jsonl",
    "neutral0":    f"{folder}/{prefix}_neutral0_generation.jsonl",
    "neutral1":    f"{folder}/{prefix}_neutral1_generation.jsonl",
    "neutral2":    f"{folder}/{prefix}_neutral2_generation.jsonl",
    "neutral5":    f"{folder}/{prefix}_neutral5_generation.jsonl",
    "neutral10":   f"{folder}/{prefix}_neutral10_generation.jsonl",
    "neutral15":   f"{folder}/{prefix}_neutral15_generation.jsonl",
    "neutral20":   f"{folder}/{prefix}_neutral20_generation.jsonl",
}
dfs = {}
for key, filepath in filepaths.items():
    try:
        dfs[key] = pd.read_json(filepath, lines=True)
    except ValueError:
        raise FileNotFoundError(f"File not found: {filepath}")
# dfs["neutral0"] = dfs["neutral0"][:200]

orig_df["embeddings"] = [emb for emb in sentenceTransformer.encode(orig_df['output'].to_list())]
for key, df in dfs.items():
    shorten = lambda text: text[:100] + text[100:].split("\n\n")[0]
    outputs = df["output"].apply(shorten)
    # sentence_only = lambda text: text.split(".")[0]
    # outputs = df["output"].apply(sentence_only)
    df["embeddings"] = [emb for emb in sentenceTransformer.encode(outputs)]
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
plt.figure(figsize=(10, 6))
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

# TSNE PLOTS OF EACH df["embeddings"]

# Combine embeddings from all dataframes
embeddings = np.vstack([df["embeddings"].tolist() for df in dfs.values()])
total_df = pd.concat([df.assign(**{'Type': key}) for key, df in dfs.items()], ignore_index=True)

# Perform t-SNE
reducer = TSNE(n_components=2)
tsne_embeddings_reduced = reducer.fit_transform(embeddings)
total_df["tsnedim1"] = tsne_embeddings_reduced[:, 0]
total_df["tsnedim2"] = tsne_embeddings_reduced[:, 1]

total_df["truncated_output"] = total_df["output"].str.slice(0, 25)
mini_df = total_df
# mini_df = total_df[total_df["Type"].isin(["transferred", "original", "neutral0"])]

fig = px.scatter(
    mini_df,
    x="tsnedim1",
    y="tsnedim2",
    labels={"tsnedim1": "TSNE dim1", "tsnedim2": "TSNE dim2"},
    hover_data=["Type", "orig_prompt", "transplant_prompt", "truncated_output"],
    color="Type",
)

fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), font_size=16)

# Save the figure as an image file instead of displaying it
pio.write_image(fig, "tsne-highres-all.png", format="png", width=750, height=450, scale=4)

# If you need to view the plot in a Jupyter notebook or similar environment,
# you can use the following line instead of fig.show():
from IPython.display import Image
Image("tsne-highres-all.png")

# %%

# Perform PHATE
phate_op = phate.PHATE(n_components=2, verbose=False)
phate_embeddings_reduced = phate_op.fit_transform(embeddings)
total_df["phatedim1"] = phate_embeddings_reduced[:, 0]
total_df["phatedim2"] = phate_embeddings_reduced[:, 1]

total_df["truncated_output"] = total_df["output"].str.slice(0, 25)
mini_df = total_df
# mini_df = total_df[total_df["Type"].isin(["transferred", "original", "neutral0"])]

fig = px.scatter(
    mini_df,
    x="phatedim1",
    y="phatedim2",
    labels={"phatedim1": "PHATE dim1", "phatedim2": "PHATE dim2"},
    hover_data=["Type", "orig_prompt", "transplant_prompt", "truncated_output"],
    color="Type",
)

fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), font_size=16)

# Save the figure as an image file instead of displaying it
pio.write_image(fig, "phate-highres-all.png", format="png", width=750, height=450, scale=4)

# If you need to view the plot in a Jupyter notebook or similar environment,
# you can use the following line instead of fig.show():
from IPython.display import Image
Image("phate-highres-all.png")

# %%
for key, df in dfs.items():
    outputs = []
    for output in df["output"].to_list():
        print({key: output[:100] + output[100:].split("\n\n")[0]})
        break

# %%
