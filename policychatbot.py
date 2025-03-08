import os
import streamlit as st
import requests
from bs4 import BeautifulSoup
import faiss
import numpy as np
import time
from mistralai.client import MistralClient
from mistralai import UserMessage

# Correct import

# ✅ Secure API Key Handling
os.environ["MISTRAL_API_KEY"] = "HVu1lheglNRREvb4XO5Yvm7GrcsufpLj"
api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    st.error("MISTRAL_API_KEY is missing. Set it as an environment variable.")
    st.stop()

client = MistralClient(api_key=api_key)

st.title("UDST Academic Policy Chatbot")

# ✅ Improved Policy Selection
option = st.selectbox(
    "Which policy would you like to inquire about?",
    [
        "academic-annual-leave-policy",
        "academic-appraisal-policy",
        "academic-appraisal-procedure",
        "academic-credentials-policy",
        "academic-freedom-policy",
        "academic-members-retention-policy",
        "academic-professional-development",
        "academic-qualifications-policy",
        "credit-hour-policy",
        "intellectual-property-policy",
    ],
)

st.write(f"You selected: {option}")

# ✅ Caching API Calls to Prevent Rate Limits
@st.cache_data(ttl=86400)  # Cache for 24 hours
def fetch_policy_text(policy_name):
    """Fetch and extract text from the UDST policy page."""
    url = f"https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/{policy_name}"
    response = requests.get(url)
    
    if response.status_code != 200:
        return None  # Return None if the page is not found

    soup = BeautifulSoup(response.text, "html.parser")
    tag = soup.find("div")
    
    return tag.text.strip() if tag else None

text = fetch_policy_text(option)

if text is None:
    st.error("Policy document not found or page structure changed.")
    st.stop()

# ✅ Chunk the text
chunk_size = 512
chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

# ✅ Caching Embeddings to Reduce API Calls
@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_text_embedding(list_txt_chunks):
    """Get embeddings with rate limit handling."""
    time.sleep(2)  # Avoid hitting API rate limit
    embeddings_batch_response = client.embeddings.create(model="mistral-embed", inputs=list_txt_chunks)
    return np.array([e.embedding for e in embeddings_batch_response.data])

# ✅ Store embeddings for document
text_embeddings = get_text_embedding(chunks)

# ✅ Create FAISS Index
d = len(text_embeddings[0])  # Dimension size
index = faiss.IndexFlatL2(d)
index.add(text_embeddings)

# ✅ User Query Input
query = st.text_input("Enter your query:")

if query:
    query_embedding = get_text_embedding([query])[0].reshape(1, -1)

    D, I = index.search(query_embedding, k=2)  # Retrieve top 2 relevant chunks
    retrieved_chunk = " ".join([chunks[i] for i in I[0]])

    # ✅ Properly formatted prompt
    prompt = f"""
    Context information is below:
    ---------------------
    {retrieved_chunk}
    ---------------------
    Based on the provided context, answer the following query:
    Query: {query}
    Answer:
    """

    def mistral(user_message, model="mistral-small-latest", is_json=False):
        model = "mistral-large-latest"
        client = Mistral(api_key=api_key)
        messages = [
        UserMessage(content=user_message),
        ]
        chat_response = client.chat.complete(
        model=model,
        messages=messages,
        )
        return chat_response.choices[0].message.content

    # ✅ Get response and display it
    response = mistral(prompt)
    st.write(response)
