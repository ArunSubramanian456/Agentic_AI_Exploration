from src.langgraphagenticai.tools.coding_tool import python_exec_tool_func
from src.langgraphagenticai.utils.code_utils import extract_code
import pandas as pd
from src.langgraphagenticai.state.state import State
import os
from langgraph.prebuilt import create_react_agent
import re

class node_data_cleaning:
    def __init__(self, model):
        """
        Initializes the node for data preparation based on recommendations from profile report
        This node is responsible for cleaning the data and preparing it for further analysis.
        """
        self.llm = model

    def run(self, state:State):
        """
        Runs the node to prepare the data
        """
        try:
            # Get file path from state and read the file
            file_path = state.get("data_file_path", None)
            if file_path is None:
                raise ValueError("File path is required for data cleaning.")   
            
            # Read the file
            df = pd.read_parquet(file_path)
            
            if df is None or df.empty:
                raise ValueError("Input data is required for cleaning.")
            
            file_path = state['data_file_path']
            parentdir_path = os.path.join(os.path.dirname(file_path),  "df_updated.parquet")

            # print(f"Parent directory: {parentdir_path}")
            profile_report = state["profile_report"]
            
            prompt = f"""
            Instructions:
            You are an expert data scientist. Review the provided profile report, write a python code to update the given dataframe.

            Your should make the following changes to the dataframe:
            - Based on the data profiling report recommendations, update the dataframe.

            Your code should :
            1. Not import any libraries
            2. Make a copy of the original dataframe to avoid modifying the original data using **X = df.copy()**.
            3. Strictly follow the recommendations provided in the profile report {profile_report}.
            4. Do not use inplace=True in any operation.


            You should double check the code to ensure the accuracy,completeness and safety of the code before executing it.
            Constraints:

            1.  **Strictly Pandas DataFrame Operations:** You must only execute code that directly manipulates Pandas DataFrames (e.g., filtering, aggregation, merging, column transformations, saving as csv, etc.).
            2.  **No Library Imports:** The code provided will *not* contain any `import` statements. The `pandas` library will be pre-loaded and accessible via the `_globals` dictionary. Assume `pd` is already available as an alias for `pandas`.
            3.  **No Network Access:** Prohibit any operations that attempt to make network requests.
            4.  **No System Calls:** Prohibit any calls to `subprocess` or `os.system`.
            5.  **No Arbitrary Code Execution:** Do not allow execution of code that is not directly related to DataFrame manipulation. This includes, but is not limited to, defining functions, classes, or executing arbitrary Python logic.

            Safety Check Procedure:

            Before finalizing *any* code, perform the following checks:

            1.  **Syntactic Check:** Ensure the code is valid Python syntax.
            2.  **Keyword Blacklist:** Scan the code for the following prohibited keywords/functions:
                * `import`
                * `open`
                * `read_csv`
                * `os`
                * `subprocess`
                * `system`
                * `shutil`
                * `request` (and related network libraries like `urllib`, `http`, `socket`)
                * `file`
                * `with` (unless part of a `df.pipe()` or similar known safe Pandas construct)
                * `exec`
                * `eval`
                * `lambda` (unless part of a known safe Pandas apply/transform operation)
                * `def`
                * `class`
            3.  **Object Access Check (Heuristic):** Favor code that operates directly on variables assumed to be DataFrames (e.g., `df.column`, `df.groupby(...), df.to_csv()`).

            **If the code passes all safety checks, finalize it and provide the output. If it fails any check, go back and update the code.**

            **Example Valid Input:**

            ```python
            df['new_col'] = df['col1'] + df['col2']
            df_filtered = df[df['col3'] > 10]
            df_grouped = df.groupby('category')['value'].mean()
            df_grouped.to_csv('./grouped_output.csv', index=False)
            ```
            **Provide only the python code as the output**.
            """

            agent = create_react_agent(model=self.llm,
                                       tools=[],
                                       prompt = prompt
                                       )
            response = agent.invoke({"messages:": [{"role": "user", "content": "Create a python code to update the dataframe based on the profile report."}]},
             {"recursion_limit":11}
             )

            result = response['messages'][-1].content
            print(result)

            code = extract_code(result)

            code_explanation = self.llm.invoke(f""" Explain this code {code}""").content

            data_cleaning_report = f"""Data Cleaning Code\n```python{code}```\n\nExplanation\n{code_explanation}"""

            df_updated = python_exec_tool_func(df, code)
            print(df_updated.info())
            
            df_updated.to_parquet(parentdir_path, index=False)

            state["data_cleaning_report"] = data_cleaning_report
          
        except Exception as e:
            print(f"Error in cleaning data: {str(e)}\n\n")
            state["error_message"].append(f"Error in cleaning data: {str(e)}\n\n")
            raise e

        return state