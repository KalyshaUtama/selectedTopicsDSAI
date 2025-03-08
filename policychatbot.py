import os
import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
import faiss
import os
from mistralai import Mistral
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import numpy as np

os.environ["MISTRAL_API_KEY"] = "b1HKlTtc2a1pqPEgMcqm8sVqckYAdfMj"
print(f"MISTRAL_API_KEY: {os.environ.get('MISTRAL_API_KEY')}")
api_key = os.getenv("MISTRAL_API_KEY")
@st.cache_data(ttl=86400)  
def get_text_embedding(list_txt_chunks):
  client = MistralClient(api_key=api_key)
  embeddings_batch_response = client.embeddings.create(model="mistral-embed",
  inputs=list_txt_chunks)
  return embeddings_batch_response.data

st.title("UDST academic policy chatbot")

option = st.selectbox(
    "Which policy would you like to inquire about?",
    (
"academic-annual-leave-policy",
"academic-appraisal-policy",
"academic-appraisal-procedure",
"academic-credentials-policy",
"academic-freedom-policy",
"academic-members’-retention-policy",
"academic-professional-development",
"academic-qualifications-policy",
"credit-hour-policy",
"intellectual-property-policy")
)

st.write(f"You selected:{option}")
response = requests.get(
f"https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/{option}"
)
html_doc = response.text
soup = BeautifulSoup(html_doc, "html.parser")
tag = soup.find("div")
text = tag.text
chunk_size = 512
chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
text_embeddings = get_text_embedding(chunks)
embeddings = np.array([text_embeddings[i].embedding for i in range(len(text_embeddings))])
d = len(text_embeddings[0].embedding)
index = faiss.IndexFlatL2(d)
index.add(embeddings)

query = st.text_input("Enter your query:")
if query:
  question_embeddings = np.array([get_text_embedding([query])[0].embedding])
  D, I = index.search(question_embeddings, k=2)
  retrieved_chunk = [chunks[i] for i in I.tolist()[0]]
  
  prompt = f"""
  Context information is below.
  ---------------------
  {retrieved_chunk}
  ---------------------
  Given the context information and not prior knowledge, answer the query.
  Query: {query}
  Answer:
  """
  
  def mistral(user_message, model="mistral-small-latest", is_json=False):
    """Query Mistral AI with a structured conversation format."""
    messages = [ChatMessage(role="user", content=user_message)]
    response = client.chat(model=model, messages=messages)
    return response.choices[0].message.content
  
  response = mistral(prompt)
  st.write(response)


          

