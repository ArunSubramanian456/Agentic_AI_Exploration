import os
import streamlit as st
from langchain_groq import ChatGroq


## LLM Model Setup
class GroqLLM:
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
            groq_api_key = self.user_controls_input.get("GROQ_API_KEY")
        
            if not groq_api_key:
                st.error("Error: Groq API Key is required.")
        
            # Initialize and return the Groq LLM model
            llm = ChatGroq(model=self.user_controls_input.get("selected_groq_model"), 
                            groq_api_key=groq_api_key, temperature = 0.1)
            
            return llm
            
        except Exception as e:
            raise Exception(f"Groq LLM init failed : {e}")