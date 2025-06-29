from typing_extensions import TypedDict, Optional, List
import pandas as pd
import src.langgraphagenticai.utils.constants as const

class State(TypedDict):
    """
    State class for managing the state of the LangGraph workflow.
    This class is used to define the structure of the state that will be passed between nodes in the graph.
    """
    next_node: str
    data_file_path: Optional[str] # file path for input data
    data_dictionary : Optional[str]  # Optional field for data dictionary
    target_metric: Optional[str]  # Optional field for target metric
    profile_report: Optional[str]
    data_cleaning_report: Optional[str]
    stats_summary_report: Optional[str]
    univariate_analysis_report: Optional[str]
    bivariate_analysis_report: Optional[str]
    final_report: Optional[str]
    error_message: List[str]


def make_initial_state() -> State:
    return {
        "next_node": "",
        "data_file_path": None,
        "data_dictionary": None,  # Optional field for data dictionary
        "target_metric": None,
        "profile_report": None,
        "data_cleaning_report": None,
        "stats_summary_report": None,
        "univariate_analysis_report": None,
        "bivariate_analysis_report": None,
        "final_report": None,
        "error_message": [],
    }