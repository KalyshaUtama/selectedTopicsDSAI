import streamlit as st
import numpy as np
import os

os.environ["MISTRAL_API_KEY"] = "HVu1lheglNRREvb4XO5Yvm7GrcsufpLj"
print(f"MISTRAL_API_KEY: {os.environ.get('MISTRAL_API_KEY')}")
api_key = os.getenv("MISTRAL_API_KEY")

import os
from mistralai import Mistral, UserMessage
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


# Streamlit UI
st.title("I am Mistral :))")

option = st.selectbox(
    "How would you like to be contacted?",
    ("Email", "Home phone", "Mobile phone"),
)

st.write("You selected:", option)
# Input query
query = st.text_input("Enter your query:")

response = mistral(query)

st.write("Response\n", response)
