from configparser import ConfigParser
import os

current_working_dir = os.path.dirname(os.path.abspath(__file__))
config_file_path = os.path.join(current_working_dir, "uiconfigfile.ini")

class Config:
    def __init__(self, config_file = config_file_path):
        self.config = ConfigParser()
        self.config.read(config_file)
    
    def get_llm_option(self):
        return self.config["DEFAULT"].get("LLM_OPTIONS").split(", ")

    def get_groq_model_options(self):
        return self.config["DEFAULT"].get("GROQ_MODEL_OPTIONS").split(", ")
    
    def get_openai_model_options(self):
        return self.config["DEFAULT"].get("OPENAI_MODEL_OPTIONS").split(", ")

    def get_page_title(self):
        return self.config["DEFAULT"].get("PAGE_TITLE")