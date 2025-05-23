# LangGraph AI SQL DB Agent 

This application allows you to connect to a SQL database and generate SQL queries from natural language. You can ask questions, and the agent will generate SQL queries to retrieve the information you need. The agent uses OpenAI's GPT-3.5-turbo model to understand your questions, generate SQL queries and provide answers.

## Features

You will discover how to build Text 2 SQL DB Agents using

- LangChain's in-built SQL Tools: LangChain's SQLDatabase and SQLDatabaseToolkit provides the agent with necessary tools to connect and query the SQL DB effectively.
- LangGraph's prebuilt ReAct Agent: Equip your LLM with ability to role play, reflect on the user query, call the right tools and execute them, analyze the tool results, and decide on the necessary next steps to arrive at the final answer.
- Streamlit UI - Create an interactive UI including page configuration, input elements, state management and response streaming.


### Running the script

- Clone the repo from GitHub using `git clone https://github.com/ArunSubramanian456/Agentic_AI_Exploration.git`
- Once you are in 'LangGraph_text2sql' folder, run `streamlit run app_v2.py` from command line
- Start asking your questions related to the Northwind database in the chat screen


## Pre requisites

- Sign up for Free LangSmith api key from https://www.langchain.com/. This is used for pull the SQL Agent Prompt Template from Hub directly.
- Sign up for OpenAI api key from https://platform.openai.com/api-keys. This is used for LLM calls.

![logo](https://github.com/ArunSubramanian456/Agentic_AI_Exploration/blob/main/LangGraph_text2sql/streamlit.png?raw=true)

