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

    def get_boxplot_data(self, df: pd.DataFrame) -> dict:
        """
        Calculates essential boxplot statistics for all numeric columns
        in a Pandas DataFrame, including quartiles, min/max, IQR, and
        whether outliers exist.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            dict: A dictionary where keys are numeric column names and values are
                dictionaries containing simplified boxplot statistics:
                {
                    'column_name': {
                        'q1': float,
                        'median': float,
                        'q3': float,
                        'min_value': float,  # Absolute minimum value in the series
                        'max_value': float,  # Absolute maximum value in the series
                        'iqr': float,
                        'outlier_exists': bool
                    },
                    ...
                }
                Returns an empty dictionary if no numeric columns are found.
        """
        boxplot_data = {}
        
        for col in df.columns.to_list():

            series = df[col]

            # Calculate quartiles and median
            q1 = series.quantile(0.25)
            median = series.quantile(0.5)
            q3 = series.quantile(0.75)
            
            iqr = q3 - q1
            
            # Calculate outlier fences
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Determine absolute min and max values of the series
            min_value = series.min()
            max_value = series.max()

            # Check if outliers exist
            lower_outliers = bool((series < lower_bound).any())
            upper_outliers = bool((series > upper_bound).any())
            
            boxplot_data[col] = {
                'q1': q1,
                'median': median,
                'q3': q3,
                'min_value': min_value,
                'max_value': max_value,
                'iqr': iqr,
                'lower_outliers': lower_outliers,
                'upper_outliers': upper_outliers,
            }
        
        return boxplot_data

    def univariate(self, df: pd.DataFrame, file_path:str):
        """
        Performs univariate analysis on the given dataframe and generates visualizations.

        Args:
            df (pd.DataFrame): Input dataframe to analyze
            file_path (str): Path to save generated visualization files

        Returns:
            tuple: A tuple containing:
                - list: List of tuples with visualization titles,file paths and data

        """
        visualizations = []
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
            if len(numeric_cols) > 0:
                if len(numeric_cols) > 3:
                    figsize_selected = (10, 3 * len(numeric_cols))
                else:
                    figsize_selected = (10,8)
                fig, axes = plt.subplots(len(numeric_cols),1,figsize = figsize_selected)
                for i, col in enumerate(numeric_cols):
                    sns.histplot(df[col], ax = axes[i], kde=True)
                    axes[i].set_title(f"Histogram of {col}")
                
                plt.tight_layout()
                path = os.path.abspath(os.path.join(imgdir_path, "Histograms.png"))
                fig.savefig(path, bbox_inches='tight')
                histogram_data = df[numeric_cols].describe().transpose().to_markdown()
                visualizations.append(("Histograms of Numeric Features", path, histogram_data))
                print(histogram_data)
                plt.close(fig)


                # Box Plots for Each Numeric Column
                fig, axes = plt.subplots(len(numeric_cols), 1, figsize=figsize_selected)
                for i, col in enumerate(numeric_cols):
                    sns.boxplot(x=df[col], ax=axes[i])
                    axes[i].set_title(f"Box Plot of {col}")
                
                plt.tight_layout()
                path = os.path.abspath(os.path.join(imgdir_path, "Boxplots.png"))
                fig.savefig(path, bbox_inches='tight')
                boxplot_data = self.get_boxplot_data(df[numeric_cols])
                boxplot_report = f"""Box Plot data of numeric features\n\n{boxplot_data}\n\n"""
                visualizations.append(("Box Plots of Numeric Features", path, boxplot_report))
                print(boxplot_report)
                plt.close(fig)
                
                # univariate_stats = df[numeric_cols].describe().transpose().to_markdown()
                # univariate_report += f"""Histogram and Box Plot data of numeric features\n\n{univariate_stats}\n\n"""

            if len(categorical_cols) > 0:
                if len(categorical_cols) > 3:
                    figsize_selected = (10, 3 * len(categorical_cols))
                else:
                    figsize_selected = (10,8)

                # Count Plots for Each Categorical Column
                fig, axes = plt.subplots(len(categorical_cols), 1, figsize=figsize_selected)
                for i, col in enumerate(categorical_cols):
                    sns.countplot(x=df[col], ax=axes[i], hue=df[col], order = df[col].value_counts().index)
                    axes[i].set_xticks(axes[i].get_xticks())
                    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=270)
                    axes[i].set_title(f"Count Plot of {col}")
                    
                plt.tight_layout()
                path = os.path.abspath(os.path.join(imgdir_path, "Countplots.png"))
                fig.savefig(path, bbox_inches='tight')
                plt.close(fig)

                countplot_report = ""  
                for col in categorical_cols:
                    countplot_data = df[col].value_counts(ascending=False).to_markdown()
                    countplot_report += f"""Count Plot data of {col}\n\n{countplot_data}\n\n"""
                print(countplot_report)
                visualizations.append(("Count Plots of Categorical Features", path, countplot_report))

        except Exception as e:
            raise Exception(f"An error occurred while generating univariate plots: {e}")
        
        return visualizations


    def run(self, state:State):
        """
        Generates a report based on univariate visualizations of the DataFrame.
        
        Returns:
            str: The complete report with visualizations and their analyses.
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
            
            # Call univariate analysis
            visualizations = self.univariate(df, str(file_path))

            data_dictionary = state.get("data_dictionary", "")

            # Setup prompt
            prompt_histograms = """"
            You are an expert data analyst. Analyze the following data {chart_data} and provide a concise report.

            **Leverage the {data_dictionary} context to incorporate any domain-specific insights.**

            The data is used to create histograms
                      
            
            Your analysis should be concise and cover the following aspects:
            - Describe the shape of the distribution (e.g., symmetric, skewed right/left, unimodal, bimodal, uniform).
            - Comment on the typical range of the data.
            - Describe the  spread or variability of the data.
            - Point out any apparent outliers or unusual features.
            - Insights on what the data might represent based on its distribution.
            - Discuss the implications of the observed patterns in the context of the data.

            Please structure your response as a Markdown report with clear headings and bullet points.
            """

            prompt_boxplots = """"

            You are an expert data analyst. Analyze the following data {chart_data} and provide a concise report.

            **Leverage the {data_dictionary} context to incorporate any domain-specific insights.**

            The data is used to create boxplots

           Your analysis should be comprehensive and cover the following aspects:
            - Describe the skewness (if any) based on the median's position compared to 25th percentile and 75th percentile.
            - Comment on the spread of the middle 50 percent of the data.
            - Comment on the overall range of the data, excluding outliers.
            - Point out any apparent outliers and their approximate locations.
            - Insights on the data might represent.
            - Discuss the implications of the observed patterns in the context of the data.

            Please structure your response as a Markdown report with clear headings and bullet points.
            """

            prompt_countplots = """
            You are an expert data analyst. Analyze the following data {chart_data} and provide a concise report.

            **Leverage the {data_dictionary} context to incorporate any domain-specific insights.**

            The data is used to create count plots

            Your analysis should be comprehensive and cover the following aspects:
            - Provide a high level summary of the categories and their relative frequencies
            - Indicate which categories represent the most and least frequent.
            Comment on the overall distribution of frequencies across categories (e.g., relatively even, heavily skewed towards one category, presence of many less frequent categories).
            - Highlight any dominant or surprisingly rare categories.
            - Insights on what these frequencies tell us about the categorical variable being analyzed.

            Please structure your response as a Markdown report with clear headings and bullet points.

            """

            # Analyze each visualization 
            report_content = "## Univariate Analysis Report\n\n"
        
            for title, img_path, data in visualizations:
                encoded_image = encode_image_to_base64(img_path)
                image_markdown_tag = f"![{title}](data:image/png;base64,{encoded_image})"
                report_content += f"### {title}\n\n{image_markdown_tag}\n\n"
                if "Histograms" in title:
                    prompt_text = prompt_histograms.format(chart_data=data, data_dictionary=data_dictionary)
                elif "Box Plots" in title:
                    prompt_text = prompt_boxplots.format(chart_data=data, data_dictionary=data_dictionary)
                elif "Count Plots" in title:
                    prompt_text = prompt_countplots.format(chart_data=data, data_dictionary=data_dictionary)
                response = self.llm.invoke(prompt_text)
                analysis_results = response.content if response else "No content returned from LLM."
                report_content += f"{analysis_results}"
            
            state["univariate_analysis_report"] = report_content
            
        except Exception as e:
            state["error_message"].append(f"Error in univariate analysis: {str(e)}\n\n")
            raise e

        return state
