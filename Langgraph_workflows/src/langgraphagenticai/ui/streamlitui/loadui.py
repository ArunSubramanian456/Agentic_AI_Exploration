import streamlit as st
from src.langgraphagenticai.state.state import make_initial_state

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from src.langgraphagenticai.ui.config import Config
import src.langgraphagenticai.utils.constants as const

class LoadStreamlitUI:
    def __init__(self):
        self.config = Config()
        self.user_controls = {}

    def initialize_session(self):
        st.session_state.stage = "START"
        st.session_state.thread_id = ""
        st.session_state.data_dictionary = ""
        st.session_state.file_path = None
        st.session_state.eda_state = make_initial_state()
        st.session_state.kpi = None

    def load_streamlit_ui(self):
        st.set_page_config(page_title=self.config.get_page_title(), page_icon=":robot_face:", layout="wide")
        st.header(self.config.get_page_title())
        

        # Sidebar for user controls
        with st.sidebar:

            ## LLM Selection
            self.user_controls['selected_llm'] = st.selectbox("Select a LLM", self.config.get_llm_option())

            if self.user_controls['selected_llm'] == 'Groq':
                self.user_controls['selected_groq_model'] = st.selectbox("Select a Model", self.config.get_groq_model_options())
                self.user_controls["GROQ_API_KEY"] = st.session_state["GROQ_API_KEY"] = st.text_input("Enter Groq API Key", type="password")

                # Validate Groq API Key
                if not self.user_controls["GROQ_API_KEY"]:
                    st.warning("Please enter your Groq API Key to proceed. Don't have one? refer : https://console.groq.com/keys")

            if self.user_controls['selected_llm'] == 'OpenAI':
                self.user_controls['selected_openai_model'] = st.selectbox("Select a Model", self.config.get_openai_model_options())
                self.user_controls["OPENAI_API_KEY"] = st.session_state["OPENAI_API_KEY"] = st.text_input("Enter OpenAI API Key", type="password")

                # Validate OpenAI API Key
                if not self.user_controls["OPENAI_API_KEY"]:
                    st.warning("Please enter your OpenAI API Key to proceed. Don't have one? refer : https://platform.openai.com/account/api-keys")


            # ## Use Case Selection
            # self.user_controls["selected_usecase"] = st.selectbox("Select a Use Case", self.config.get_usecase_options())

            if st.button("Reset Session"):
                # for key in list(st.session_state.keys()):
                #     del st.session_state[key]

                self.initialize_session()
                st.session_state.file_uploader_key += 1

                st.rerun()
            

        return self.user_controls
    

    





            





