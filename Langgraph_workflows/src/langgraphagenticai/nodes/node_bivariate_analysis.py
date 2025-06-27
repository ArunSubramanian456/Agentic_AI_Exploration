import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from tabulate import tabulate
import os
import numpy as np
from src.langgraphagenticai.utils.encode_image_to_base64 import encode_image_to_base64
from src.langgraphagenticai.state.state import State

class node_bivariate_analysis:
    def __init__(self, model):
        """
        Initializes the node for bivariate analysis.
        This node is responsible for generating the bivariate analysis report.
        """
        self.llm = model

    def bivariate(self, df: pd.DataFrame, file_path:str):

        visualizations = []
        bivariate_report = f""
        parentdir_path = os.path.dirname(file_path) # route to TEMP_DATA_DIR
        imgdir_path = os.path.join(parentdir_path, "images") # this directory should already exists as we created it during univariate analysis

        # Get numeric cols and categorical cols
        numeric_cols = df.select_dtypes(include=np.number).columns.to_list()
        categorical_cols = [col for col in df.select_dtypes(include=['object', 'category']) if df[col].nunique() < 20]

        try:
            # Correlation chart for Numeric Columns
            if len(numeric_cols) > 1:
                plt.figure(figsize=(12, 8))
                corr = df[numeric_cols].corr()
                sns.heatmap(corr, annot=True, fmt=".2f", cmap='Spectral',vmin=-1,vmax=1)
                plt.title("Correlation Heatmap of Numeric Features")
                path = os.path.abspath(os.path.join(imgdir_path, "Correlation_heatmap.png"))
                plt.savefig(path, bbox_inches='tight')
                visualizations.append(("Correlation Heatmap of Numeric Features", path))
                plt.close();
            
                corr_report = df[numeric_cols].corr().to_markdown()
                bivariate_report +=f"Correlation Heatmap data\n\n{corr_report}\n\n"

            # Box Plots for Categorical Columns against Numeric Columns
            if numeric_cols and categorical_cols:
                for cat_col in categorical_cols:
                    for num_col in numeric_cols:
                        plt.figure(figsize=(10, 6))
                        sns.boxplot(x=df[cat_col], 
                                    y=df[num_col], 
                                    hue = df[cat_col], 
                                    legend = False,
                                    order=df.groupby(cat_col, observed=False)[num_col].median().sort_values(ascending=False).index
                                    )
                        plt.title(f"Box Plot of {num_col} by {cat_col}")
                        plt.xlabel(cat_col)
                        plt.ylabel(num_col)
                        if df[cat_col].nunique() > 10:
                            plt.xticks(rotation=270)
                        path = os.path.abspath(os.path.join(imgdir_path, f"Boxplot_{num_col}_vs_{cat_col}.png"))
                        plt.savefig(path, bbox_inches='tight')
                        visualizations.append((f"Box Plot of {num_col} by {cat_col}", path))
                        plt.close();

                        
                        bivariate_stats = df.groupby(cat_col, observed=False)[num_col].describe().transpose()
                        bivariate_report_md = tabulate(bivariate_stats, headers='keys', tablefmt='pipe')
                        bivariate_report += f"""Box Plot data of {num_col} by {cat_col}\n\n{bivariate_report_md}\n\n"""

        except Exception as e:
            print(f"An error occurred while generating bivariate plots: {e}")
        
        return visualizations, bivariate_report


    def run(self, state:State):
        """
        Generates a report based on bivariate visualizations of the DataFrame.
        
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
            visualizations, bivariate_report = self.bivariate(df, str(file_path))

            # Setup prompt
            prompt = f"""
            You are an expert data analyst. Analyze the following data {bivariate_report} and provide a detailed report.

            The data is used to create visualizations such as histograms, box plots, 
            count plots, correlation charts, pair plots, or bivariate box plots.
            
            Your analysis should be comprehensive and cover the following aspects:
            - Provide insights based on the characteristic of the data for each plot.
            - Discuss the implications of the observed patterns in the context of the data.
            
            The analysis should be structured based on the type of chart the data supports.
            
            For Correlation Chart data, please provide:
            - Discuss the perceived strength and direction (positive/negative) of the correlation between key pairs of variables.
            - Highlight any pairs exhibiting strong direct or inverse relationships.
            - Point out any variables that show little to no apparent relationship.
            - Insights on what these relationships might imply about the connections between the variables.

            For **Bivariate Box Plot data**, please provide:
            - Compare the distributions of the numeric variable across different categories:
                - Discuss  differences in central tendencies (50%) between groups
                - Discuss how the spread (75% minus 50%) varies between groups
                - Describe any apparent relationships between the categorical variable and the numeric variable.
            - Provide insights on how the categorical variable might influence the numeric variable.

            Please structure your response as a Markdown report with clear headings and bullet points.
            """

            # Analyze each visualization using the vision_analysis function
            report_content = "## Bivariate Analysis Report\n\n"
        
            for title, img_path in visualizations:
                encoded_image = encode_image_to_base64(img_path)
                image_markdown_tag = f"![{title}](data:image/png;base64,{encoded_image})"
                report_content += f"### {title}\n\n{image_markdown_tag}\n\n"

            response = self.llm.invoke(prompt)
            analysis_results = response.content if response else "No content returned from LLM."
            report_content += f"{analysis_results}"
        
            state["bivariate_analysis_report"] = report_content
            
        except Exception as e:
            state["error_message"].append(f"Error in bivariate analysis: {str(e)}\n\n")
            raise e

        return state
