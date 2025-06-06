# Q&A Chatbot using LangChain

This project explores the use of LangChain framework to build a Q&A chatbot trained on Attention Mechanism using "Attention is all you need" research paper from arxiv.org

## Features

You will discover

- How to set up Two-tier architecture - Streamlit for Presentation tier and FASTAPI for Application tier
- How to build a RAG enabled Q&A Chatbot using LangChain
- How to deploy your GenAI solution as an API using FastAPI


## Running the script

- Clone the repo from GitHub using `git clone https://github.com/ArunSubramanian456/Agentic_AI_Exploration.git`
- Once you are in 'LangChain_RAG_Chatbot' folder, host the server using `python server.py` from command line
- Test the API by accessing Swagger UI in `http://127.0.0.1:8000/docs` 
- Open another command line and host the client using `streamlit run client.py` to run the Streamlit app
- Start asking questions to the chatbot


## Credits

This exploratory project is based on the learnings from this [Udemy Course](https://www.udemy.com/course/complete-generative-ai-course-with-langchain-and-huggingface)


## Pre requisites

- Sign up for Free LangChain api key from https://www.langchain.com/. This is used for LangSmith Tracing.
- Sign up for Free Groq AI API from https://groq.com/. This gives you access to open source foundation models (llama, gemma, mixtral) with very fast inference
- Sign up for Free Hugging Face account and get an access token from https://huggingface.co/


![Application tier](https://github.com/ArunSubramanian456/Agentic_AI_Exploration/blob/main/LangChain_RAG_Chatbot/server.png?raw=true)

![Presentation tier](https://github.com/ArunSubramanian456/Agentic_AI_Exploration/blob/main/LangChain_RAG_Chatbot/client.png?raw=true)

