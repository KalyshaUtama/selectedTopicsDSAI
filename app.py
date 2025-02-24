import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load precomputed document embeddings
embeddings = np.load("embeddings.npy")

# Load documents
with open("documents.txt", "r", encoding="utf-8") as f:
    documents = [line.strip() for line in f.readlines()]  # Strip newlines

def retrieve_top_k(query_embedding, embeddings, k=10):
    """Retrieve top-k most similar documents using cosine similarity."""
    similarities = cosine_similarity(query_embedding.reshape(1, -1), embeddings)  # Shape: (1, num_docs)
    similarities = similarities.flatten()  # Convert to 1D array
    top_k_indices = similarities.argsort()[-k:][::-1]  # Get top-k indices
    return [(documents[i], similarities[i]) for i in top_k_indices]

# Streamlit UI
st.title("Information Retrieval using Document Embeddings")

# Input query
query = st.text_input("Enter your query:")

# Load or compute query embedding (Placeholder)
def get_query_embedding(query):
    return np.random.rand(embeddings.shape[1])  # Replace with actual embedding model

if st.button("Search"):
    query_embedding = get_query_embedding(query)
    results = retrieve_top_k(query_embedding, embeddings)

    # Display results
    st.write("### Top 10 Relevant Documents:")
    for doc, score in results:
        st.write(f"- **{doc}** (Score: {score:.4f})")
