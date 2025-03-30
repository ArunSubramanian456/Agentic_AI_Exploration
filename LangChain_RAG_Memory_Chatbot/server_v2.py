# Importing the required libraries

import os
import shutil
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
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

# Requirements for Chat History
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

# Requirements for FastAPI backend
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from langchain import globals

# Requirement for Data Models
from pydantic import BaseModel

# Requirement for managing Chroma DB documents
from uuid import uuid4

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

# Temp directory for storing files
temp_dir = os.path.join(os.path.dirname(__file__), "temp")
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)


## LLM Model Setup
llm=ChatGroq(model="Gemma2-9b-It",groq_api_key=groq_api_key) 
llm_parsed = llm | StrOutputParser()


## Prompt Template

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])


qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Use the following context to answer the user's question."),
    ("system", "Context: {context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

vectorstore=None # variable to hold the vector store.
sessionstore = {} # variable to hold the user session data
prev_ids  = [] # variable to hold the previous ids.


# Function to get session history
def get_session_history(user_id:str) -> BaseChatMessageHistory:
    """
    Get the session history for the user ID.
    """
    if user_id not in sessionstore:
        sessionstore[user_id] = ChatMessageHistory()
    return sessionstore[user_id]


# Function to process the PDF file
def process_pdf(file_path:str):
    """
    Process the PDF file and create a retrieval-augmented generation (RAG) chain."
    """
    try:
        # Reset the session store when a new PDF is processed
        # Ideally this should be done in a more secure way, like using a database or a cache store
        # but for simplicity, we are using a dictionary here.
        global sessionstore
        if len(sessionstore) > 0:
            sessionstore = {} # variable to hold the user session data

        # Initialize the user ID for the session
        user_id = "default" 

        # document loader
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        # text splitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        final_documents = text_splitter.split_documents(documents)

        # Get the uuids for the documents
        curr_ids = [str(uuid4()) for _ in range(len(final_documents))]
        # print(f"UUIDs: {curr_ids}") #Debug

        # embeddings
        embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        ## Vector Store
        global prev_ids
        global vectorstore
        if vectorstore is not None:
            vectorstore.delete(ids = prev_ids)  # Remove the previous documents from the vector store
        
        vectorstore=Chroma.from_documents(documents=final_documents,
                                          embedding=embeddings, 
                                          ids = curr_ids
                                          )
        retriever=vectorstore.as_retriever()

        # Create the retrieval chain
        history_aware_retriever = create_history_aware_retriever(llm_parsed, retriever, contextualize_q_prompt)
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        prev_ids = curr_ids # Store the current ids for the next call.

        conversation_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        return conversation_rag_chain

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


rag_chain_instance = None # Global variable to hold the rag chain.


# Create the FastAPI app
app = FastAPI(title="Langchain Server",
            version="1.0",
            description="A simple API server using Langchain runnable interfaces")


@app.post("/process_pdf")
async def process_pdf_endpoint(file: UploadFile = File(...)):
    """
    Endpoint to process the uploaded PDF file and create a retrieval-augmented generation (RAG) chain.
    """
    try:
        # Save the uploaded file to the temp directory
        
        file_path = os.path.join(temp_dir, file.filename)
        if os.path.exists(file_path):
            os.remove(file_path)
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Process the PDF and create the RAG chain
        global rag_chain_instance
        if rag_chain_instance is not None:
            rag_chain_instance = None  # Reset the previous instance if it exists
        rag_chain_instance = process_pdf(file_path)

        return JSONResponse(content={"message": "PDF processed successfully!"})

    except Exception as e:
        print(f"Error in /process_pdf/: {e}") #Debug
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}") # Raise an HTTP exception.

# Setup Data Validation using Pydantic
class InvokeRequest(BaseModel):
    input: str  # The input field, as expected by your LangChain chain

@app.post("/invoke")
async def invoke(request: InvokeRequest):
    try:
        global rag_chain_instance
        if rag_chain_instance is None:
            raise HTTPException(status_code=400, detail="No PDF processed. Please upload a PDF first.")
        
        data = request.model_dump()
        result = rag_chain_instance.invoke(data, config = {"configurable": {"session_id" : "default"}})
        print(f"{sessionstore}") #Debug
        answer = result['answer']
        sources = [doc.page_content for doc in result['context']] #extract page contents from documents

        return JSONResponse(content={"answer": answer, "sources": sources})

    except Exception as e:
        print(f"Error in /invoke: {e}") #Debug
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}") # Raise an HTTP exception.


@app.get("/", response_class=HTMLResponse)
async def welcome():
    with open("index.html", "r") as f:
        html_content = f.read()
    return html_content


if __name__=="__main__":
    import uvicorn
    uvicorn.run("__main__:app",host="127.0.0.1",port=8000, reload=True)