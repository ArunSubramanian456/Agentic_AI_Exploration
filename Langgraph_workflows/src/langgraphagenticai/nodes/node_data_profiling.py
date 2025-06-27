import pandas as pd
import io
from src.langgraphagenticai.state.state import State

class node_data_profiling:
    def __init__(self, model):
        """
        Initializes the node for profile input data.
        This node is responsible for generating the data profile report from input data.
        """
        self.llm = model

    def run(self, state:State):
        """
        Runs the node to generate profile report
        """
        try:
            # Get file path from state and read the file
            file_path = state.get("data_file_path", None)
            if file_path is None:
                raise ValueError("File path is required for profiling.")   
            
            # Read the file and store it in the state
            df = pd.read_parquet(file_path)
            
            if df is None or df.empty:
                raise ValueError("Input data is required for profiling.")
            
            buf = io.StringIO()
            df.info(buf=buf)
            metadata = buf.getvalue()
            head = df.head().to_markdown(index=False)
            missing = df.isnull().sum()
            missing = missing[missing > 0]
            missing_info = "No missing values." if missing.empty else str(missing)

            content = f"""
            ## Data Profiling Report\n
            ### Metadata\n
            ```{metadata}```\n\n
            ### Preview\n
            {head}\n\n
            ### Missing Values\n
            {missing_info}\n
            """

            prompt = f"""
            You are a data profiling expert. Analyze the following data and provide a detailed report.
            Your report should include the following details:
            - Number of rows and columns
            - Data types of each column
            - Sample of the first few rows
            - Recommendations for handling duplicates, if applicable.
              Duplicate records should be removed or handled appropriately.
            - Recommend the right data types for each column, if applicable.
              For example, entity IDs that represents person, location, product should be strings, 
              dates should be in datetime format, metric columns like sales should be numeric, ranking, scores and similar ordinal data should be categorical.
            - Recommendations for handling missing values, if applicable.
            - If a column has more than 60% missing values, it should be dropped.
            - For a numeric column with less than 60% missing values, it should be filled with the mean or median.
            - For a categorical column with less than 60% missing values, it should be treated as a separate category or filled with the mode.
            - For time series data with less than 60% missing values, missing values should be filled with forward-fill or backward-fill
            - When missing value signifies a distinct state or category (eg. 'Does Not Exist', 'No Event Occured'), impute with sentinel values like 0 or -1

            Here is the data you need to analyze:
            ```{content}```
            """

            response = self.llm.invoke(prompt)
            state["profile_report"] = response.content if response else "No content returned from LLM."
            
        except Exception as e:
            state["error_message"].append(f"Error in profiling input data: {str(e)}\n\n")
            raise e

        return state
