import os
import streamlit
import requests
from bs4 import BeautifulSoup
import re
import faiss

os.environ["MISTRAL_API_KEY"] = "HVu1lheglNRREvb4XO5Yvm7GrcsufpLj"
print(f"MISTRAL_API_KEY: {os.environ.get('MISTRAL_API_KEY')}")
api_key = os.getenv("MISTRAL_API_KEY")


def add_document(doc):
  response = requests.get(
  doc
  )
  html_doc = response.text
  soup = BeautifulSoup(html_doc, "html.parser")
  tag = soup.find("div")
  text = tag.text
  print(text)
  chunk_size = 512
  chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)
  client = Mistral(api_key=api_key)
  text_embeddings = client.embeddings.create(model="mistral-embed",
  inputs=list_txt_chunks).data
  embeddings = np.array([text_embeddings[i].embedding for i in range(len(text_embeddings))])
  d = len(text_embeddings[0].embedding)
  index = faiss.IndexFlatL2(d)
  index.add(embeddings)

st.title("UDST policy chatbot")

option = st.selectbox(
    "Which policy would you like to inquire about?",
    (
"Academic Appraisal Policy -V1	PL-AC-09",
"Academic Appraisal Procedure-V1 	PR-AC-09",
"Academic Credentials Policy - V1 	PL-AC-02",
"Academic Freedom Policy- V1 ",
"Academic Members' Retention Policy- V1 	PL-AC-12",
"Academic Professional Development Policy-V1",
"Academic Qualifications Policy -V2	PL-AC-03",
"Credit Hour Policy-V1	PL-AC-26",
"Intellectual Property Policy -V1",
"Joint Appointment Policy - V1)"
)

st.write("You selected:", option)


          

