import os
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pathlib import Path
from langchain import globals

globals.set_verbose(True)  # To turn on verbosity

# Load the environment variables
load_dotenv()

## Langsmith Tracking
os.environ["LANGSMITH_TRACING"]=os.getenv("LANGSMITH_TRACING")
os.environ["LANGSMITH_ENDPOINT"]=os.getenv("LANGSMITH_ENDPOINT")
os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"]=os.getenv("LANGSMITH_PROJECT")


## Prompt Template
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please respond to the question asked"),
        ("user","Question:{question}")
    ]
)

## Ollama Llama3.2 model
llm=ChatOllama(model="llama3.2", temperature = 0.5, num_predict = 256)
output_parser=StrOutputParser()
chain=prompt|llm|output_parser

## Streamlit framework
st.title("Langchain Demo With Llama 3.2 Model")
st.subheader("Chat with the model")
input_text=st.text_input("Enter your question here:")

if input_text:
    with st.spinner("Generating answer..."):
        answer = chain.invoke({"question":input_text})
    st.subheader("Answer:")
    st.write(answer)


