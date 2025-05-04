import os
import io
from typing import Annotated, TypedDict 
from dotenv import load_dotenv
import json
from IPython.display import display, Image
from PIL import Image as PILImage


# Tools and Agents
from langgraph.graph import  StateGraph, START, END
from langgraph.graph.message import add_messages 
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

# Data Models and FastAPI
from pydantic import BaseModel
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

# Load environment variables
load_dotenv()

## Langsmith Tracking
os.environ["LANGSMITH_TRACING"]=os.getenv("LANGSMITH_TRACING")
os.environ["LANGSMITH_ENDPOINT"]=os.getenv("LANGSMITH_ENDPOINT")
os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"]=os.getenv("LANGSMITH_PROJECT")
groq_api_key = os.getenv("GROQ_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")
os.environ["TAVILY_API_KEY"] = tavily_api_key
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
os.environ["USER_AGENT"] = "MyLangChainApp/1.0"


# LLM setup
# llm=ChatGroq(model="Gemma2-9b-It",groq_api_key=groq_api_key) 
llm = ChatOpenAI(api_key=OPENAI_API_KEY, 
                 model="gpt-3.5-turbo", 
                 temperature=0, 
                 ) 



# Tools setup
@tool
def internet_search(query: str):
    """
    Search the web for realtime and latest information.
    for examples, news, stock market, weather updates etc.
    
    Args:
    query: The search query
    """
    search = TavilySearchResults(
        max_results=2
    )

    response = search.invoke(query)

    return response

@tool
def llm_search(query: str):
    """
    Use the LLM model for general and basic information.
    """
    response = llm.invoke(query)
    return response

tools = [internet_search, llm_search]


# Bind the tools to the LLM
llm_with_tools = llm.bind_tools(tools)
# llm_with_tools


# Define a state dictionary to hold the message history
class State(TypedDict):
    # {"messages": ["your message"]}
    messages: Annotated[list, add_messages]

# Define the chatbot function to be used as a node in the graph
def chatbot(state: State):
    print(state["messages"])
    response = llm_with_tools.invoke(state["messages"])
    print(response)
    return {"messages": [response]}

# Define the memory checkpointer
memory = MemorySaver()

# Define the graph structure with nodes and edges
graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges("chatbot", tools_condition)

graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile(checkpointer=memory)

# Save the graph image to a file
# image_bytes = graph.get_graph().draw_mermaid_png()
# image = PILImage.open(io.BytesIO(image_bytes))
# image.save("langgraph.png") 


# Create the FastAPI app
app = FastAPI(title="Langgraph Server",
            version="1.0",
            description="A simple API server using Langgraph")


origins = ["http://localhost:8501", "http://127.0.0.1:8501"]  # Streamlit
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class UserInput(BaseModel):
    message: str
    # To manage conversation history
    thread_id: str  


@app.post("/chat")
async def chat_endpoint(user_input: UserInput):
    """
    Endpoint to receive user input and process it with the LangGraph.
    """
    thread_id = user_input.thread_id
    config = {"configurable": {"thread_id": thread_id}}
    inputs = {"messages": [user_input.message]}
    results = await graph.ainvoke(inputs, config)
    return {"response": results["messages"][-1].content}


@app.get("/", response_class=HTMLResponse)
async def welcome():
    with open("index.html", "r") as f:
        html_content = f.read()
    return html_content


if __name__=="__main__":
    import uvicorn
    uvicorn.run("__main__:app",host="127.0.0.1",port=8000, reload=True)
