import os
import streamlit as st
from langchain_openai import ChatOpenAI


## LLM Model Setup
class OpenAILLM:
    """
    Class to configure and return the Groq LLM model based on user input.
    """
    def __init__(self, user_controls_input):
        self.user_controls_input = user_controls_input

    def get_llm_model(self):
        """
        Returns the configured Groq LLM model based on user input.
        """
        try:
            openai_api_key = self.user_controls_input.get("OPENAI_API_KEY")
        
            if not openai_api_key:
                st.error("Error: OpenAI API Key is required.")
        
            # Initialize and return the OpenAI LLM model
            llm = ChatOpenAI(model=self.user_controls_input.get("selected_openai_model"), 
                             api_key=openai_api_key, temperature = 0.1)
            
            return llm
            
        except Exception as e:
            raise Exception(f"OpenAI LLM init failed : {e}")