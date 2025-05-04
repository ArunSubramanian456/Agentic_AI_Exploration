import streamlit as st
import requests
import json

# Function to get response from the server
def get_openai_response(message, thread_id):
    json_body={"message": message, "thread_id": thread_id}
    headers = {'Content-Type': 'application/json'}  # Add Content-Type header
    response=requests.post("http://127.0.0.1:8000/chat", json=json_body, headers=headers)

    return response.json()

## Streamlit app
st.set_page_config(page_title="LangGraph AI Agent Chat", page_icon="💬", layout="wide")
st.title("LangGraph AI Agent Chat")

with st.sidebar:
    thread_id = st.text_input("Enter a user session id", value="default")

message = st.chat_input("Enter your question ")


if message and thread_id:
    with st.spinner("Generating answer..."):
        answer_data = get_openai_response(message, thread_id)

    st.write(answer_data["response"])
