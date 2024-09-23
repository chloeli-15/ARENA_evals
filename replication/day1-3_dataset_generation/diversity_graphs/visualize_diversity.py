#%%
import plotly.express as px
from utils import import_json
import sentence_transformers
from sentence_transformers import SentenceTransformer
import umap
import pandas as pd
import hdbscan
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import plotly.io as pio

#%%#
# =================== Sentence Embeddings ===================

# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# The sentences to encode
dataset_names = ["scores/quality/2c-generated-balanced.json"]
datasets = []
for name in dataset_names:
    dataset = import_json(name)
    datasets.extend(dataset)

questions = [item["question"] for item in datasets]
match_id = [item["answer_matching_behavior"][0] for item in datasets]
answer_matching_behavior = [item["answers"][m] for item,m in zip(datasets, match_id)]
labels = [item["label"] for item in datasets]
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
umap_embeddings = umap.UMAP(n_neighbors=20, # constraining the size of the local neighborhood UMAP will look at when attempting to learn the manifold structure of the data; low values of n_neighbors will force UMAP to concentrate on very local structure (potentially to the detriment of the big picture); large values will push UMAP to look at larger neighborhoods of each point, losing fine detail structure for the sake of getting the broader of the data
                            n_components=2, # dimensionality of the reduced dimension space; 2D for 2D plot
                            min_dist=0.0, # minimum distance apart that points are allowed to be; determines how tightly the embedding can be packed
                            metric='cosine').fit_transform(embeddings)

# 2. Cluster the reduced dimension embeddings
# HDBSCAN is a density-based clustering algorithm: it groups together points that are closely packed together (points with many neighbors), marking as outliers points that lie alone in low-density regions
clustering = hdbscan.HDBSCAN(min_cluster_size=15, # smallest size grouping that you wish to consider a cluster
                          metric='euclidean',                      
                          cluster_selection_method='eom').fit(umap_embeddings)

# 3. Create a DataFrame for visualization
result = pd.DataFrame(umap_embeddings, columns=['x', 'y'])
result['labels'] = clustering.labels_
result['question'] = questions
result['answer_matching_behavior'] = answer_matching_behavior
result['label'] = labels

# 4. # Separate outliers and clustered data
outliers = result[result.labels == -1]
clustered = result[result.labels != -1]

# ======================= Topic Modeling =======================
# Topic modeling is a type of statistical modeling for discovering the abstract "topics" that occur in a collection of documents. 
# TF-IDF (Term Frequency-Inverse Document Frequency) is a statistical measure that evaluates how relevant a word is to a document in a collection of documents.

# 1. Create a single document for each cluster
docs_df = pd.DataFrame(questions, columns=["Doc"])
docs_df['Topic'] = clustering.labels_
docs_df['Doc_ID'] = range(len(docs_df))
docs_per_topic = docs_df.groupby(['Topic'], as_index = False).agg({'Doc': ' '.join})

# 2. Create a TF-IDF Vectorizer

def c_tf_idf(documents, m, ngram_range=(1, 1)):
    """
    t = frequency of each word extracted for each class i
    w = sum of all words in class i
    m = total unjoined number of documents
    n = total number of classes
    """
    count = CountVectorizer(ngram_range=ngram_range, stop_words="english").fit(documents) # CountVectorizer: Convert a collection of text documents to a matrix of token counts
    t = count.transform(documents).toarray() # Transform documents to document-term matrix
    w = t.sum(axis=1) # Sum of all words in each class
    tf = np.divide(t.T, w) # Term frequency
    sum_t = t.sum(axis=0) #
    idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
    tf_idf = np.multiply(tf, idf)

    return tf_idf, count
  
tf_idf, count = c_tf_idf(docs_per_topic.Doc.values, m=len(questions))

# 3. Extract top keywords from each cluster
def extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20):
    words = count.get_feature_names_out()
    labels = list(docs_per_topic.Topic)
    tf_idf_transposed = tf_idf.T
    indices = tf_idf_transposed.argsort()[:, -n:]
    top_n_words = {label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in enumerate(labels)}
    return top_n_words

def extract_topic_sizes(df):
    topic_sizes = (df.groupby(['Topic'])
                     .Doc
                     .count()
                     .reset_index()
                     .rename({"Topic": "Topic", "Doc": "Size"}, axis='columns')
                     .sort_values("Size", ascending=False))
    return topic_sizes

top_n_words = extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20)
topic_sizes = extract_topic_sizes(docs_df); topic_sizes.head(10)

# Plotting
fig = px.scatter(
    result, x='x', y='y', color='labels',
    color_continuous_scale='hsv_r',
    title='Power-seeking Dataset Topics ',
    labels={'x': 'UMAP 1', 'y': 'UMAP 2', 'labels': 'Cluster'},
    hover_data={'question': True, 'answer_matching_behavior': True,  'label': True, 'x': False, 'y': False, 'labels': False}
)

# Add outliers to the plot
fig.add_scatter(x=outliers['x'], y=outliers['y'], mode='markers', marker=dict(color='#BDBDBD', size=5), name='Outliers')

# Adding top words as annotations

for label, top_words in top_n_words.items():
    if label != -1:  # Do not annotate outliers
        cluster_center = result[result.labels == label][['x', 'y']].mean()
        words_annotation = '<br>'.join([word for word, _ in top_words[:5]])
        fig.add_annotation(
            x=cluster_center['x'], y=cluster_center['y'], text=words_annotation,
            showarrow=True, arrowhead=1, ax=0, ay=-30,
            bordercolor=None, borderwidth=0, borderpad=0, bgcolor='rgba(0,0,0,0)', opacity=1
        )
fig.update_layout(
    width=1200,  # Set the desired width
    height=800,   # Set the desired height
    plot_bgcolor='white',  # Set plot background color to white
    paper_bgcolor='white',  # Set paper background color to white
    xaxis=dict(showgrid=False, zeroline=False, visible=False),  # Hide x-axis
    yaxis=dict(showgrid=False, zeroline=False, visible=False)   # Hide y-axis
)

# Custom hover template with wrapped text
def wrap_text(text, width=45):
    return '<br>'.join([text[i:i+width] for i in range(0, len(text), width)])

result['wrapped_question'] = result['question'].apply(wrap_text)
result['wrapped_answer'] = result['answer_matching_behavior'].apply(wrap_text)
result['wrapped_labels'] = result['label'].astype(str)


hover_template = (
    '<b>Question:</b><br>%{customdata[0]}<br>'
    '<b>Power-seeking answer:</b><br>%{customdata[1]}<br>'
    '<b>Label:</b> %{customdata[2]}<br>'
    '<extra></extra>'  # Disable default trace name in hover
)

fig.update_traces(
    hovertemplate=hover_template,
    customdata=np.stack((result['wrapped_question'], result['wrapped_answer'], result['wrapped_labels']), axis=-1),
    marker=dict(size=9)  # Adjust the size of the markers if needed
)

# Enable zooming and panning
config = {
    'scrollZoom': True,  # Enable zooming with trackpad
    'displayModeBar': True,  # Show the mode bar with zoom and pan options
    'modeBarButtonsToAdd': ['zoomIn2d', 'zoomOut2d', 'pan2d', 'resetScale2d']  # Add specific buttons
}
fig.show(config=config)

#%%
# Export the figure as an HTML file
#THIS EXPORTS THE FILE AS AN HTML. MAKE SURE FILEPATH DOES NOT OVERWRITE EXISTING FILE BEFORE RUNNING!
# pio.write_html(fig, file='umap_clustering_7clus.html', auto_open=True, config=config) 
# Plot scores fo 4-generated questions vs 8-generated questions

