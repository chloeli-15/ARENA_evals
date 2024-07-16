#%%
import plotly.express as px
from utils import import_json
from sentence_transformers import SentenceTransformer

#%%
# Diversity UMPA 


# Load a pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# The sentences to encode
dataset_name = ""
dataset = import_json(dataset_name)
sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
]

# 2. Calculate embeddings by calling model.encode()
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# 3. Calculate the embedding similarities
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.6660, 0.1046],
#         [0.6660, 1.0000, 0.1411],
#         [0.1046, 0.1411, 1.0000]])

# %%
# Plot score for 5-shots vs 10-shots

# Plot scores fo 4-generated questions vs 8-generated questions
