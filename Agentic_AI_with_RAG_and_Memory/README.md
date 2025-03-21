# AI Agent with RAG and memory

This project explores the capabilities of Phidata (an open source agentic AI framework) and Groq (an open source AI inference engine) to build a AI Agent that is capable of Retrieval Augmented Generation (RAG) as well as utilizing stored memory from chat history.

## Features

You will discover

- How to use Phidata to create AI Agents
- How to host a POSTGRESQL database with PgVector extension from Docker Containers for RAG search
- How to equip the AI Agent with PDFUrlKnowledgeBase to provide recommendations on Thai food recipes
- How to equip the AI Agent with memory storage to utilize historic chat data and context while providing responses
- How to equip the AI Agent with generic LLM capabilities using Llama 3

## Running the script

- To interact with your AI Agent in command line interface, execute ` python pdf_agent.py`.

## Credits

This exploratory projects are built following tutorials found on this [YouTube channel](https://www.youtube.com/@krishnaik06)


## Pre requisites

- Sign up for Free and get a Groq api key from https://groq.com/ . This is used by the AI Agent for LLM inference. 
- Download and install Docker Desktop. This will be used to host a POSTGRESQL database with pgvector extension for similarity search
> - Alternatively, you can use other options like ChromaDB, PineCone,etc.. It will help you skip the Docker installation and use if you are new to it.

## Screenshots

### Agent RAG use

Here you can notice that the Agent is utilizing data from a [Thai Recipes PDF file](https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf) stored in knowledge base to provide response

![logo](https://github.com/ArunSubramanian456/Agentic_AI_Exploration/blob/main/Agentic_AI_with_RAG_and_Memory/agent_rag.png?raw=true)

### Agent Memory use

Here you can see how Agent recalls memory from previous chat history 

![logo](https://github.com/ArunSubramanian456/Agentic_AI_Exploration/blob/main/Agentic_AI_with_RAG_and_Memory/agent_memory_retrieval.png?raw=true)

### Agent LLM use

Here you can see how Agent utilizes its own training data to make (outdated) movie recommendations to go with Pad Thai.

![logo](https://github.com/ArunSubramanian456/Agentic_AI_Exploration/blob/main/Agentic_AI_with_RAG_and_Memory/agent_llm.png?raw=true)