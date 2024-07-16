#%%
import plotly.express as px
from utils import import_json
import sentence_transformers
from sentence_transformers import SentenceTransformer
import umap
import pandas as pd
import hdbscan

#%%#
# =================== Sentence Embeddings ===================

# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# The sentences to encode
dataset_names = ["datasets/2c-generated-4.json", "datasets/2c-generated-5.json", "datasets/2c-generated-6.json", "datasets/2c-generated-7.json"]
datasets = []
for name in dataset_names:
    dataset = import_json(name)
    datasets.extend(dataset)

questions = [item["question"] for item in datasets]

#%%
# 2. Calculate embeddings by calling model.encode()
embeddings = model.encode(questions, show_progress_bar=True)
print(embeddings.shape)
# [3, 384]

# 3. Calculate the embedding similarities
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.6660, 0.1046],
#         [0.6660, 1.0000, 0.1411],
#         [0.1046, 0.1411, 1.0000]])

# %%
# ====================== UMAP Clustering ======================

# 1. Reduce the dimensionality of the embeddings
umap_embeddings = umap.UMAP(n_neighbors=15, 
                            n_components=2, # dimensionality of the reduced dimension space; 2D for 2D plot
                            min_dist=0.0, # minimum distance apart that points are allowed to be; determines how tightly the embedding can be packed
                            metric='cosine').fit_transform(embeddings)

# 2. Cluster the reduced dimension embeddings
clustering = hdbscan.HDBSCAN(min_cluster_size=10, # smallest size grouping that you wish to consider a cluster
                          metric='euclidean',                      
                          cluster_selection_method='eom').fit(umap_embeddings)

# 3. Create a DataFrame for visualization
result = pd.DataFrame(umap_embeddings, columns=['x', 'y'])
result['labels'] = clustering.labels_

# 4. # Separate outliers and clustered data
outliers = result[result.labels == -1]
clustered = result[result.labels != -1]

# 5. Visualize clusters using Plotly
fig = px.scatter(
    result, x='x', y='y', color='labels',
    color_continuous_scale='hsv_r',
    title='UMAP Clustering',
    labels={'x': 'UMAP 1', 'y': 'UMAP 2', 'labels': 'Cluster'}
)

# Add outliers to the plot
fig.add_scatter(x=outliers['x'], y=outliers['y'], mode='markers', marker=dict(color='#BDBDBD', size=5), name='Outliers')


fig.show()

# Plot scores fo 4-generated questions vs 8-generated questions

# %%
