# Importing the required libraries

import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq


from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from langchain import globals

from pydantic import BaseModel

globals.set_verbose(True)  # To turn on verbosity

# Load the environment variables
load_dotenv()

## Langsmith Tracking
os.environ["LANGSMITH_TRACING"]=os.getenv("LANGSMITH_TRACING")
os.environ["LANGSMITH_ENDPOINT"]=os.getenv("LANGSMITH_ENDPOINT")
os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"]=os.getenv("LANGSMITH_PROJECT")
groq_api_key=os.getenv("GROQ_API_KEY")
os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
os.environ["USER_AGENT"] = "MyLangChainApp/1.0"

## Document Loader
loader=PyPDFLoader('attention.pdf')
docs=loader.load()


## Text Splitter
text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
final_documents=text_splitter.split_documents(docs)


## Embeddings
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

## Vector Store
vectorstore=Chroma.from_documents(documents=final_documents,embedding=embeddings)
retriever=vectorstore.as_retriever()

## LLM Model Setup
llm=ChatGroq(model="Gemma2-9b-It",groq_api_key=groq_api_key) 
llm_parsed = llm | StrOutputParser()


## Prompt Template
system_prompt = (
    "You are a helpful AI assistant capable of handling question & answer tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. For any other irrelevant question, answer that you are specialized only in Attention mechanism"
    " and cannot answer questions that are not related to Attention mechanism. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt=ChatPromptTemplate.from_messages(
    [
        ("system",system_prompt),
        ("user","{input}")
    ]
)


# Create the retrieval chain
question_answer_chain=create_stuff_documents_chain(llm_parsed,prompt)
rag_chain=create_retrieval_chain(retriever,question_answer_chain)

# Testing the chatbot
# response=rag_chain.invoke({"input":"What is the purpose of masked multihead attention layer in decoder?"}) 
# print(response['answer'])


# Setup Data Validation using Pydantic
class InvokeRequest(BaseModel):
    input: str  # The input field, as expected by your LangChain chain


# Create the FastAPI app
app = FastAPI(title="Langchain Server",
            version="1.0",
            description="A simple API server using Langchain runnable interfaces")

# add_routes(app, rag_chain)

@app.post("/invoke")
async def invoke(request: InvokeRequest):
    try:
        data = request.model_dump()
        # print(f"Received data: {data}") #Debug

        result = rag_chain.invoke(data)
        # print(f"Langchain result: {result}") #Debug

        answer = result['answer']
        sources = [doc.page_content for doc in result['context']] #extract page contents from documents

        return JSONResponse(content={"answer": answer, "sources": sources})

    except Exception as e:
        print(f"Error in /invoke: {e}") #Debug
        raise HTTPException(status_code=500, detail=str(e)) # Raise an HTTP exception.

@app.get("/", response_class=HTMLResponse)
async def welcome():
    with open("index.html", "r") as f:
        html_content = f.read()
    return html_content


if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host="127.0.0.1",port=8000)