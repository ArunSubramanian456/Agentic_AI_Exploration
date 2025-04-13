import os
from typing import Any
from dotenv import load_dotenv
import json


# Tools and Agents
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor,  Tool, create_react_agent
from langchain_core.agents import AgentAction
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper, GoogleSerperAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
from langchain import hub

# Data Models and FastAPI
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

## Langsmith Tracking
os.environ["LANGSMITH_TRACING"]=os.getenv("LANGSMITH_TRACING")
os.environ["LANGSMITH_ENDPOINT"]=os.getenv("LANGSMITH_ENDPOINT")
os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"]=os.getenv("LANGSMITH_PROJECT")
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
os.environ["USER_AGENT"] = "MyLangChainApp/1.0"


# LLM setup
llm = ChatOpenAI(api_key=OPENAI_API_KEY, 
                 model="gpt-3.5-turbo", 
                 temperature=0, 
                 streaming=True, 
                 ) 


# Tools setup
arxiv_wrapper = ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=1000)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=1000)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

# search = GoogleSerperAPIWrapper()
search = DuckDuckGoSearchRun()

tools = [
    Tool(name="Search", func=search.run, description="Useful for general web searches."),
    Tool(name="Wikipedia", func=wiki.run, description="Useful for looking up information on Wikipedia."),
    Tool(name="Arxiv", func=arxiv.run, description="Useful for searching academic papers on arXiv."),
]

# Define the prompt template for the Deep Learning tutor
prompt = hub.pull("hwchase17/react")

prompt.template = """Answer the following questions in detail as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: a detailed response to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""

# Define the Pydantic model for the request body
class ChatRequest(BaseModel):
    query: str

app = FastAPI(title="Langchain Server", version="1.0", description="A simple API server using Langchain")

origins = ["http://localhost:8501", "http://127.0.0.1:8501"]  # Streamlit
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

async def run_agent(query: str):
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent =agent,
        llm=llm,
        tools=tools,
        prompt=prompt,
        verbose=True,
    )
    async for event in agent_executor.astream({"input": query}):
        # yield f"data: {event}\n\n"
        
        if "actions" in event:
            for action in event["actions"]:
                if isinstance(action, AgentAction):
                    yield f"{json.dumps({'type': 'agent_action', 'tool': action.tool, 'tool_input': action.tool_input, 'log': action.log})}\n\n"
        elif "output" in event:
            yield f"{json.dumps({'type': 'final_answer', 'output': event['output']})}\n\n"

        elif "error" in event:
            yield f"{json.dumps({'type': 'error', 'error': str(event['error'])})}\n\n"

    yield f"{json.dumps({'type': 'end'})}\n\n"

@app.post("/stream_chat")
async def stream_chat(chat_request: ChatRequest):
    query = chat_request.query
    if not query:
        return {"error": "Missing query"}
    return StreamingResponse(run_agent(query), media_type="text/event-stream")


@app.get("/", response_class=HTMLResponse)
async def welcome():
    with open("index.html", "r") as f:
        html_content = f.read()
    return html_content

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("__main__:app", host="127.0.0.1", port=8000, reload=True)