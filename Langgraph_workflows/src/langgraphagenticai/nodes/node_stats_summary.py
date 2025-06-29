import pandas as pd
from tabulate import tabulate   
from src.langgraphagenticai.state.state import State
import os

class node_stats_summary:
    def __init__(self, model):
        """
        Initializes the node for stats summary report.
        This node is responsible for generating stats summary report from cleaned data.
        """
        self.llm = model
    def run(self, state:State):
        """
        Runs the node to generate stats summary report.
        """
        try:
            # Get file path from state and read the file
            file_path = state.get("data_file_path", None)
            parentdir_path = os.path.join(os.path.dirname(file_path), "df_updated.feather")

            if parentdir_path is None:
                raise ValueError(f"{parentdir_path} does not exist.")   
            
            # Read the file and store it in the state
            df = pd.read_feather(parentdir_path)
            
            if df is None or df.empty:
                raise ValueError("Input data is required for profiling.")
            
            summary_stats = df.describe(include='all').transpose().to_markdown()
            # summary_stats_tab = tabulate(summary_stats, headers='keys', tablefmt='pipe')

            data_dictionary = state.get("data_dictionary", "")

            # Summarize the observations from the summary statistics
            prompt = f"""
            You are a data analyst. Analyze the following summary statistics of the DataFrame utilizing the context in data dictionary and provide a concise report.
            Your report should include the following details:
            - Summary statistics data table
            - Observations on the distribution of numeric columns
            - Observations on the distribution of categorical columns

            **Leverage the data dictionary context to incorporate any domain-specific insights.**

            Here is the summary statistics you need to analyze:
            ```{summary_stats}```

            Here is the data dictionary you need to understand the context of the data:
            ```{data_dictionary}```
            """
            response = self.llm.invoke(prompt)
            results = response.content if response else "No content returned from LLM."

            stats_summary_report = f"""
            ## Summary Statistics Analysis Report\n\n
            {results}\n\n 
            """
            state["stats_summary_report"] = stats_summary_report
            
        except Exception as e:
            state["error_message"] = f"Error in summary stats report: {str(e)}"
            raise e

        return state
