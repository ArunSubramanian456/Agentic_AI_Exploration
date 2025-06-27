import streamlit as st
import json
from src.langgraphagenticai.ui.streamlitui.loadui import LoadStreamlitUI
from src.langgraphagenticai.LLMS.groqllm import GroqLLM
from src.langgraphagenticai.LLMS.openllm import OpenAILLM
from src.langgraphagenticai.state.state import State
from src.langgraphagenticai.graph.graph_builder import GraphBuilder
from src.langgraphagenticai.graph.graph_executor import GraphExecutor
from src.langgraphagenticai.utils.clean_directory import clean_directory
import src.langgraphagenticai.utils.constants as const

import os
import numpy as np
import pandas as pd
import shutil

# MAIN Function START
def load_langgraph_agenticai_app():
    """
    Main entry point for the Streamlit application.
    This function initializes the Streamlit UI, loads user input, configures the LLM,
    and sets up the LangGraph workflow. It also handles the file upload and data processing.
    """
   
    # Load UI
    ui = LoadStreamlitUI()

    if 'stage' not in st.session_state:
        ui.initialize_session()
        st.session_state.file_uploader_key = 0

    user_input = ui.load_streamlit_ui()

    if not user_input:
        st.error("Error: Failed to load user input from the UI.")
        return

    
    # Upload the input data file
    
    st.header("Data Upload and Validation")
            
    uploaded_file = st.file_uploader(
        "Upload your data file (CSV format)",
        type=['csv'],
        help = "Upload a CSV file to explore the data quality",
        key = st.session_state.file_uploader_key,
    )

    # delete any existing csv files in temp_data
    TEMP_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'temp_data'))

    # Save the new uploaded file
    if uploaded_file is not None:

        # Configure LLM
        selectedLLM = user_input.get("selected_llm")
        if selectedLLM == "Groq":
                obj_llm_config = GroqLLM(user_controls_input=user_input)
                model = obj_llm_config.get_llm_model()
        else:
            obj_llm_config = OpenAILLM(user_controls_input=user_input)
            model = obj_llm_config.get_llm_model()

        if not model:
            st.error("Error: LLM model could not be initialized.")
            return


        # Make sure temp data directory is empty
        # if it doesn't exist already, create it
        if st.session_state.stage == "START":
            clean_directory(TEMP_DATA_DIR)

        # Read in the uploaded file as a pandas dataframe
        df = pd.read_csv(uploaded_file)
        
        # Create a file path
        file_path = st.session_state.file_path = os.path.join(TEMP_DATA_DIR, "df.parquet")

        # Save as parquet file
        df.to_parquet(file_path, index=False)


        ### Graph Builder
        graph_builder=GraphBuilder(model)

        try:
            graph = graph_builder.setup_graph()
            graph_executor = GraphExecutor(graph)

        except Exception as e:
            st.error(f"Error: Graph setup failed - {e}")
            return
        
        # Initialize graph. Get file path into the AgentState
        if st.session_state.stage == "START" and st.session_state.file_path is not None:
            with st.spinner("Initializing Graph..."):
                graph_response = graph_executor.initialize_graph(st.session_state.file_path)
            st.session_state.thread_id = graph_response["thread_id"]
            st.session_state.eda_state = graph_response["eda_state"]
            st.session_state.stage = const.PROFILE_DATA
            st.rerun()

        # Profile the data
        if st.session_state.stage == const.PROFILE_DATA:
            # st.write(st.session_state.eda_state)
            if st.button("Profile Data"):
                with st.spinner("Profiling Data..."):
                    graph_response = graph_executor.graph_execution(st.session_state.thread_id, st.session_state.eda_state, const.PROFILE_DATA)
                st.session_state.eda_state = graph_response["eda_state"]
                st.session_state.stage = const.CLEAN_DATA
                st.rerun()
        
        # Clean the data
        if st.session_state.stage == const.CLEAN_DATA:
            # st.write(st.session_state.eda_state)
            st.write(st.session_state.eda_state["profile_report"])
            if st.button("Clean Data"):
                with st.spinner("Cleaning Data"):
                    graph_response = graph_executor.graph_execution(st.session_state.thread_id, st.session_state.eda_state, const.CLEAN_DATA)
                st.session_state.eda_state = graph_response["eda_state"]
                st.session_state.stage = const.SUMMARIZE_DATA
                st.rerun() 

        # Summarize the data
        if st.session_state.stage == const.SUMMARIZE_DATA:
            # st.write(st.session_state.eda_state)
            st.write(st.session_state.eda_state["data_cleaning_report"])
            if st.button("Summarize Data"):
                with st.spinner("Summarizing Data"):
                    graph_response = graph_executor.graph_execution(st.session_state.thread_id, st.session_state.eda_state, const.SUMMARIZE_DATA)
                st.session_state.eda_state = graph_response["eda_state"]
                st.session_state.stage = const.GENERATE_UNIVARIATE_REPORT
                st.rerun()
        
        # Generate Univariate analysis report
        if st.session_state.stage == const.GENERATE_UNIVARIATE_REPORT:
            # st.write(st.session_state.eda_state)
            st.write(st.session_state.eda_state["stats_summary_report"])
            if st.button("Generate Univariate Report"):
                with st.spinner("Generating Univariate Report"):
                    graph_response = graph_executor.graph_execution(st.session_state.thread_id, st.session_state.eda_state, const.GENERATE_UNIVARIATE_REPORT)
                st.session_state.eda_state = graph_response["eda_state"]
                st.session_state.stage = const.GENERATE_BIVARIATE_REPORT
                st.rerun()

        # Generate bivariate analysis report
        if st.session_state.stage == const.GENERATE_BIVARIATE_REPORT:
            # st.write(st.session_state.eda_state)
            st.write(st.session_state.eda_state["univariate_analysis_report"])
            if st.button("Generate Bivariate Report"):
                with st.spinner("Generating Bivariate Report"):
                    graph_response = graph_executor.graph_execution(st.session_state.thread_id, st.session_state.eda_state, const.GENERATE_BIVARIATE_REPORT)
                st.session_state.eda_state = graph_response["eda_state"]
                st.session_state.stage = const.GENERATE_FINAL_REPORT
                st.rerun()

        # Generate final recommendations
        if st.session_state.stage == const.GENERATE_FINAL_REPORT:
            # st.write(st.session_state.eda_state)
            st.write(st.session_state.eda_state["bivariate_analysis_report"])
            if st.button("Generate Final Recommendations"):
                with st.spinner("Generating Final Recommendations"):
                    graph_response = graph_executor.graph_execution(st.session_state.thread_id, st.session_state.eda_state, const.GENERATE_FINAL_REPORT)
                st.session_state.eda_state = graph_response["eda_state"]
                st.session_state.stage = const.END_NODE
                st.rerun()

        # Download full analyis report
        if st.session_state.stage == const.END_NODE:
            # st.write(st.session_state.eda_state)
            # Display each report in its own container
            with st.expander("Summary Statistics"):
                st.write(st.session_state.eda_state.get("stats_summary_report", "No summary statistics available."))

            with st.expander("Univariate Analysis"):
                st.write(st.session_state.eda_state.get("univariate_analysis_report", "No univariate report available."))

            with st.expander("Bivariate Analysis"):
                st.write(st.session_state.eda_state.get("bivariate_analysis_report", "No bivariate report available."))

            with st.expander("Recommendations", expanded=True):
                st.write(st.session_state.eda_state.get("final_report", "No final report available."))
        
            # Prepare full report as Markdown
            full_report_md = (
                "# Full Data Analysis Report\n\n"
                f"{st.session_state.eda_state["stats_summary_report"]}\n\n"
                f"{st.session_state.eda_state["univariate_analysis_report"]}\n\n"
                f"{st.session_state.eda_state["bivariate_analysis_report"]}\n\n"
                f"{st.session_state.eda_state["final_report"]}\n"
            )
        
            st.download_button(
                label="Download Full Data Analysis Report(Markdown)",
                data=full_report_md,
                file_name="full_report.md",
                mime="text/markdown",
            )