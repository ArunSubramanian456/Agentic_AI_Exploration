# RAG + Memory Q&A Chatbot using LangChain

This project explores the use of LangChain framework to build RAG + Memory Q&A Chatobt to answer questions related to the document users upload.

## Features

You will discover

-   How to set up a two-tier architecture: Streamlit for the presentation tier and FastAPI for the application tier.
-   How to build a RAG + Memory enabled Q&A Chatbot using LangChain.
-   How to deploy your GenAI solution as an API using FastAPI.
-   How to manage document uploads and maintain context within a chatbot application.
-   How to clear and manage vector store data efficiently.
-   How to use session state in Streamlit for a better user experience.
-   How to implement robust error handling in both the FastAPI backend and Streamlit frontend.

## Running the script

- Clone the repo from GitHub using `git clone https://github.com/ArunSubramanian456/Agentic_AI_Exploration.git`
- Once you are in 'LangChain_RAG_Memory_Chatbot' folder, host the server using `python server_v2.py` from command line
- Test the API by accessing Swagger UI in `http://127.0.0.1:8000/docs` 
- Open another command line and host the client using `streamlit run client_v2.py` to run the Streamlit app
- Upload your doc and click "Process PDF"
- Start asking questions to the chatbot and click "Generate Response"

## Credits

This exploratory project is based on the learnings from this [Udemy Course](https://www.udemy.com/course/complete-generative-ai-course-with-langchain-and-huggingface)

## Pre requisites

- Sign up for Free LangChain api key from https://www.langchain.com/. This is used for LangSmith Tracing.
- Sign up for Free Groq AI API from https://groq.com/. This gives you access to open source foundation models (llama, gemma, mixtral) with very fast inference
- Sign up for Free Hugging Face account and get an access token from https://huggingface.co/


![Application tier](https://github.com/ArunSubramanian456/Agentic_AI_Exploration/blob/main/LangChain_RAG_Memory_Chatbot/server.png?raw=true)

![Presentation tier](https://github.com/ArunSubramanian456/Agentic_AI_Exploration/blob/main/LangChain_RAG_Memory_Chatbot/client.png?raw=true)

