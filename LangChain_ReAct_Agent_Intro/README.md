# ReAct AI Agent Chatbot using LangChain

This project explores the techniques to build a ReAct AI Agent Chatbot that streams thoughts, actions and responses to user questions.

## Features

You will discover

-   How to set up a two-tier architecture: Streamlit for the presentation tier and FastAPI for the application tier.
-   How to build a ReAct AI Agent Chatbot using LangChain and provide it access to Search, Wikipedia and Arxiv tools.
-   How to stream the thoughts, actions and responses of the ReAct agent using FastAPI
-   How to request and receive the streaming content and display it in Streamlit frontend
-   How to implement robust error handling in both the FastAPI backend and Streamlit frontend.

## Running the script

- Clone the repo from GitHub using `git clone https://github.com/ArunSubramanian456/Agentic_AI_Exploration.git`
- Once you are in 'LangChain_ReAct_Agent_Intro' folder, host the server using `python server_4.py` from command line
- Test the API by accessing Swagger UI in `http://127.0.0.1:8000/docs` 
- Open another command line and host the client using `streamlit run client.py` to run the Streamlit app
- Start asking questions to the chatbot

## Credits

This exploratory project is based on the learnings from this [Udemy Course](https://www.udemy.com/course/complete-generative-ai-course-with-langchain-and-huggingface)

## Pre requisites

- Sign up for Free LangChain api key from https://www.langchain.com/. This is used for LangSmith Tracing.
- How to get OpenAI API key - https://www.youtube.com/watch?v=OB99E7Y1cMA


![Application tier](https://github.com/ArunSubramanian456/Agentic_AI_Exploration/blob/main/LangChain_ReAct_Agent_Intro/server.png?raw=true)

![Presentation tier](https://github.com/ArunSubramanian456/Agentic_AI_Exploration/blob/main/LangChain_ReAct_Agent_Intro/client.png?raw=true)

