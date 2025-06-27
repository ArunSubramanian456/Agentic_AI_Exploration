import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from tabulate import tabulate
import os
import numpy as np
from src.langgraphagenticai.utils.clean_directory import clean_directory
from src.langgraphagenticai.utils.encode_image_to_base64 import encode_image_to_base64
from src.langgraphagenticai.state.state import State

class node_univariate_analysis:
    def __init__(self, model):
        """
        Initializes the node for univariate analysis.
        This node is responsible for generating the univariate analysis report.
        """
        self.llm = model

    def univariate(self, df: pd.DataFrame, file_path:str):
        """
        Performs univariate analysis on the given dataframe and generates visualizations.

        Args:
            df (pd.DataFrame): Input dataframe to analyze
            file_path (str): Path to save generated visualization files

        Returns:
            tuple: A tuple containing:
                - list: List of tuples with visualization titles and file paths
                - str: Report containing univariate statistics
        """
        visualizations = []
        univariate_report = f""
        parentdir_path = os.path.dirname(file_path) # route to TEMP_DATA_DIR
        imgdir_path = os.path.join(parentdir_path, "images")

        # Check if directory exists; if it exists, then clean it
        # If directory doesn't exist, then create it
        clean_directory(imgdir_path)

        # Get numeric cols and categorical cols
        numeric_cols = df.select_dtypes(include=np.number).columns.to_list()
        categorical_cols = [col for col in df.select_dtypes(include=['object', 'category']) if df[col].nunique() < 20]

        try:
            # Histograms for Each Numeric Column
            fig, axes = plt.subplots(len(numeric_cols),1,figsize = (10, 8))
            for i, col in enumerate(numeric_cols):
                sns.histplot(df[col], ax = axes[i], kde=True)
                axes[i].set_title(f"Histogram of {col}")
            
            plt.tight_layout()
            path = os.path.abspath(os.path.join(imgdir_path, "Histograms.png"))
            fig.savefig(path, bbox_inches='tight')
            visualizations.append(("Histograms of Numeric Features", path))
            plt.close(fig)


            # Box Plots for Each Numeric Column
            fig, axes = plt.subplots(len(numeric_cols), 1, figsize=(10, 8))
            for i, col in enumerate(numeric_cols):
                sns.boxplot(x=df[col], ax=axes[i])
                axes[i].set_title(f"Box Plot of {col}")
            
            plt.tight_layout()
            path = os.path.abspath(os.path.join(imgdir_path, "Boxplots.png"))
            fig.savefig(path, bbox_inches='tight')
            visualizations.append(("Box Plots of Numeric Features", path))
            plt.close(fig)

            # Count Plots for Each Categorical Column
            fig, axes = plt.subplots(len(categorical_cols), 1, figsize=(10, 8))
            for i, col in enumerate(categorical_cols):
                sns.countplot(x=df[col], ax=axes[i], hue=df[col], order = df[col].value_counts().index)
                axes[i].set_xticks(axes[i].get_xticks())
                axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=270)
                axes[i].set_title(f"Count Plot of {col}")
                
            plt.tight_layout()
            path = os.path.abspath(os.path.join(imgdir_path, "Countplots.png"))
            fig.savefig(path, bbox_inches='tight')
            visualizations.append(("Count Plots of Categorical Features", path))
            plt.close(fig)
            
            univariate_stats = df[numeric_cols].describe().transpose().to_markdown()
            # univariate_report_tab = tabulate(univariate_stats, headers='keys', tablefmt='pipe')
            univariate_report += f"""Histogram and Box Plot data of numeric features\n\n{univariate_stats}\n\n"""

            for col in categorical_cols:
                univariate_stats = df[col].value_counts(ascending=False).to_markdown()
                univariate_report += f"""Count Plot data of {col}\n\n{univariate_stats}\n\n"""

        except Exception as e:
            print(f"An error occurred while generating univariate plots: {e}")
        
        return visualizations, univariate_report


    def run(self, state:State):
        """
        Generates a report based on univariate visualizations of the DataFrame.
        
        Returns:
            str: The complete report with visualizations and their analyses.
        """
        try:
            # Get file path from state and read the file
            file_path = state.get("data_file_path", None)
            parentdir_path = os.path.join(os.path.dirname(file_path), "df_updated.parquet")

            if parentdir_path is None:
                raise ValueError(f"{parentdir_path} does not exist.")   
            
            # Read the file and store it in the state
            df = pd.read_parquet(parentdir_path)
            
            if df is None or df.empty:
                raise ValueError("Input data is required for profiling.")
            
            # Call univariate analysis
            visualizations, univariate_report = self.univariate(df, str(file_path))

            # Setup prompt
            prompt = f"""
            You are an expert data analyst. Analyze the following data {univariate_report} and provide a detailed report.

            The data is used to create visualizations such as histograms, box plots, 
            count plots, correlation charts, pair plots, or bivariate box plots.
            
            Your analysis should be comprehensive and cover the following aspects:
            - Provide insights based on the characteristic of the data for each plot.
            - Discuss the implications of the observed patterns in the context of the data.
            
            The analysis should be structured based on the type of chart the data supports.
            
            For Histogram data, please provide:
            - Describe the shape of the distribution (e.g., symmetric, skewed right/left, unimodal, bimodal, uniform).
            - Comment on the typical range of the data.
            - Describe the  spread or variability of the data.
            - Point out any apparent outliers or unusual features.
            - Insights on what the data might represent based on its distribution.

            For Box Plot data, please provide:
            - Describe the skewness (if any) based on the median's position compared to 25th percentile and 75th percentile.
            - Comment on the spread of the middle 50% of the data.
            - Comment on the overall range of the data, excluding outliers.
            - Point out any apparent outliers and their approximate locations.
            - Insights on the data might represent.

            For Count Plot data, please provide:
            - Provide a high level summary of the categories and their relative frequencies
            - Indicate which categories represent the most and least frequent.
            Comment on the overall distribution of frequencies across categories (e.g., relatively even, heavily skewed towards one category, presence of many less frequent categories).
            - Highlight any dominant or surprisingly rare categories.
            - Insights on what these frequencies tell us about the categorical variable being analyzed.

            Please structure your response as a Markdown report with clear headings and bullet points.

            """

            # Analyze each visualization 
            report_content = "## Univariate Analysis Report\n\n"
        
            for title, img_path in visualizations:
                encoded_image = encode_image_to_base64(img_path)
                image_markdown_tag = f"![{title}](data:image/png;base64,{encoded_image})"
                report_content += f"### {title}\n\n{image_markdown_tag}\n\n"

            response = self.llm.invoke(prompt)
            analysis_results = response.content if response else "No content returned from LLM."
            report_content += f"{analysis_results}"
            state["univariate_analysis_report"] = report_content
            
        except Exception as e:
            state["error_message"].append(f"Error in univariate analysis: {str(e)}\n\n")
            raise e

        return state
