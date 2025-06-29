from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL
from langgraph.prebuilt import ToolNode
import pandas as pd



def coding_tool(df:pd.DataFrame, output_path:str):
    """
        Return the Python REPL tool for preparing data
    """
    python_repl = PythonREPL(_globals={"df": df, "pd": pd, "winsorize":winsorize, "output_path":output_path})
    repl_tool = Tool(name="python_repl",
                     description="A Python shell. Use this to execute python commands. Input should be a valid python command.",
                     func=python_repl.run,
                     )

    return [repl_tool]


def python_exec_tool_func(df:pd.DataFrame, code: str) -> pd.DataFrame:
    """
    Creates a Python execution tool for arbitrary code.
    """
    # Execute snippet
    namespace = dict(pd=pd, df=df)
    try:
        exec(code, {}, namespace)
    except Exception as e:
        return f"Error executing the code snippet : {e}"

    result = namespace.get("X")
    return result
