# from IPython.display import display, Image
import os
from pathlib import Path

from langchain_community.utilities import SQLDatabase
from langsmith import Client
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

# from PIL import Image as PILImage
# import io

import streamlit as st
import sqlite3
from sqlalchemy import create_engine

# STREAMLIT APP SETUP

TITLE = "Your SQL Database Agent"
st.set_page_config(page_title=TITLE, page_icon="📊", )
st.title(TITLE)



st.markdown("""
Welcome to the SQL Database Agent. This AI agent is designed to help you query Northwind SQL database and answer your questions.
You can ask questions about the database, and the agent will generate SQL queries to retrieve the information you need. 
The agent uses OpenAI's GPT-3.5-turbo model to understand your questions and generate SQL queries.
To learn more about the Northwind database, you can check out the [Northwind Database Documentation](https://docs.yugabyte.com/preview/sample-data/northwind/)
""")

st.markdown("""If you are interested in the code, visit [my GitHub repo](https://github.com/ArunSubramanian456/Agentic_AI_Exploration/blob/main/LangGraph_text2sql/)""")

with st.expander("Example Questions", expanded=False):
    st.write(
        """
        - What tables exist in the database?
        - What are the first 5 rows in the Customers table?
        - Which cities have the most customers?
        - Which customers are located in London?
        - Who are the customers with most orders?
        - Which employee has processed the most orders?
        - Show me the sales performance of each employee (e.g., total order value).
        - What are the top products with most orders?
        - What is the average price of products by category?
        - Who are the top suppliers of Teatime Chocolate Biscuits with lowest unit price?
        - What is the average dollar value (calculated as unit price * quantity) of an order?
        - What are the minimum and maximum order dates?
        - List the products supplied by suppliers located in the USA
        - Which customers ordered products from the 'Seafood' category?
        """
    )


# Setup database

@st.cache_resource(ttl="2h")
def configure_db():
    dbfilepath=(Path(__file__).parent/"northwind.db").absolute()
    print(dbfilepath)
    creator = lambda: sqlite3.connect(f"file:{dbfilepath}?mode=ro", uri=True)
    return SQLDatabase(create_engine("sqlite:///", creator=creator))

st.session_state["db"] = configure_db()
    
# db_path = os.path.join(os.path.dirname(__file__), "northwind.db")
#st.session_state["db"] = SQLDatabase.from_uri("sqlite:///{db_path}")

# STREAMLIT APP SIDEBAR

if 'OPENAI_API_KEY' not in st.session_state:
    st.session_state['OPENAI_API_KEY'] = None
if 'LANGSMITH_API_KEY' not in st.session_state:
    st.session_state['LANGSMITH_API_KEY'] = None
if 'llm' not in st.session_state:
    st.session_state['llm'] = None
if 'db' not in st.session_state:
    st.session_state['db'] = None
if 'prompt' not in st.session_state:
    st.session_state['prompt'] = None
if 'tools' not in st.session_state:
    st.session_state['tools'] = None
if 'agent_executor' not in st.session_state:
    st.session_state['agent_executor'] = None

st.sidebar.header("Enter your OpenAI API Key")

st.session_state["OPENAI_API_KEY"] = st.sidebar.text_input("OpenAI API Key", type="password", help="Your OpenAI API key is required for the app to function.")

# Test OpenAI API Key
if st.session_state["OPENAI_API_KEY"]:
    try:
        llm = ChatOpenAI(api_key=st.session_state["OPENAI_API_KEY"], 
                         model="gpt-3.5-turbo", 
                         temperature=0.5, 
                         streaming=True, 
                         ) 
        st.session_state["llm"] = llm
        st.sidebar.success("OpenAI API Key is valid.")
    except Exception as e:
        st.sidebar.error(f"Invalid OpenAI API Key: {e}")

else:
    st.sidebar.warning("Please enter your OpenAI API key to use the app.")



st.sidebar.header("Enter your LangSmith API Key")

st.session_state["LANGSMITH_API_KEY"] = st.sidebar.text_input("LangSmith API Key", type="password", help="Your Langsmith API key is required for the app to function.")

# Test Langsmith API Key
if st.session_state["LANGSMITH_API_KEY"]:
    try:
        client = Client(api_key=st.session_state["LANGSMITH_API_KEY"])
        prompt = client.pull_prompt("langchain-ai/sql-agent-system-prompt")
        st.sidebar.success("LangSmith API Key is valid.")
    except Exception as e:
        st.sidebar.error(f"Invalid LangSmith API Key: {e}")

else:
    st.sidebar.warning("Please enter your LangSmith API key to use the app.")

# st.sidebar.write("Built with ♥ by Arun")


# Setup the SQL toolkit
if st.session_state["llm"] and st.session_state["db"]:
    toolkit = SQLDatabaseToolkit(db=st.session_state["db"], llm=st.session_state["llm"])
    st.session_state["tools"] = toolkit.get_tools()


# Create the agent
if st.session_state["LANGSMITH_API_KEY"]:
    st.session_state["prompt"] = prompt.format(dialect = st.session_state["db"].dialect, top_k = 5)

if st.session_state["llm"] and st.session_state["tools"] and st.session_state["prompt"]:
    st.session_state["agent_executor"] = create_react_agent(model = st.session_state["llm"], tools = st.session_state["tools"], state_modifier=st.session_state["prompt"])

st.sidebar.markdown(f"""<div class="footer" style="">Made with<span style="font-size:150%;color:red;"> &hearts; </span>by Arun</div>""", unsafe_allow_html=True)


message = st.chat_input("Enter your question: ")

if (message 
    and st.session_state["llm"] 
    and st.session_state["db"] 
    and st.session_state["LANGSMITH_API_KEY"]
    and st.session_state["agent_executor"]):

    with st.spinner("Generating answer..."):
        st.chat_message("user").write(message)

        query = {"messages": [HumanMessage(message)]}   
        response = st.session_state["agent_executor"].stream(query, stream_mode="updates")
        
        st.session_state.full_response = ""
        st.chat_message("ai")
        report_container = st.empty()  # Create an empty container to update the report

        for step in response:
            if "agent" in step and "messages" in step["agent"]:
                for message in step["agent"]["messages"]:
                    if hasattr(message, "content") and message.content:
                        st.session_state.full_response += f"**Agent Response:** {message.content}\n\n"
                    elif hasattr(message, "tool_calls") and message.tool_calls:
                        for tool_call in message.tool_calls:
                            st.session_state.full_response += f"**Agent is calling Tool** '{tool_call['name']}' with arguments '{tool_call['args']}'\n\n"
            elif "tools" in step and "messages" in step["tools"]:
                for message in step["tools"]["messages"]:
                    if hasattr(message, "content") and message.content:
                        st.session_state.full_response += f"**Tool Result:** '{message.content}' from tool '{message.name}'\n\n"
        
            report_container.markdown(st.session_state.full_response)