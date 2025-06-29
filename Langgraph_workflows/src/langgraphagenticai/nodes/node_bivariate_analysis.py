import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from tabulate import tabulate
import os
import numpy as np
from src.langgraphagenticai.utils.encode_image_to_base64 import encode_image_to_base64
from src.langgraphagenticai.state.state import State
from pandas.api.types import is_numeric_dtype

class node_bivariate_analysis:
    def __init__(self, model):
        """
        Initializes the node for bivariate analysis.
        This node is responsible for generating the bivariate analysis report.
        """
        self.llm = model

    def get_boxplot_data(self, df: pd.DataFrame, numeric_col: str, categorical_col:str) -> dict:
        """
        Returns a dictionary with boxplot data for the given numeric and categorical columns.
        
        Args:
            df (pd.DataFrame): The DataFrame containing the data.
            numeric_col (str): The name of the numeric column.
            categorical_col (str): The name of the categorical column.
        
        Returns:
            dict: A dictionary with boxplot data.
        """

        bivariate_boxplot_data = {}

        # Get unique categories
        # Convert to list to iterate over
        categories = df[categorical_col].unique().tolist()

        for category in categories:
            series = df[df[categorical_col] == category][numeric_col]

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

            bivariate_boxplot_data[category] = {
                'q1': q1,
                'median': median,
                'q3': q3,
                'min_value': min_value,
                'max_value': max_value,
                'iqr': iqr,
                'lower_outliers': lower_outliers,
                'upper_outliers': upper_outliers,
            }

        return bivariate_boxplot_data

    def bivariate(self, df: pd.DataFrame, file_path:str, kpi:str):

        visualizations = []
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
                plt.close();
                
                corr_data = df[numeric_cols].corr().to_markdown()
                corr_report = f"Correlation Heatmap data\n\n{corr_data}\n\n"
                visualizations.append(("Correlation Heatmap of Numeric Features", path, corr_report))    

            # Box Plots for Categorical KPI against Numeric Columns
            if df[kpi].dtype in ['object', 'category']: 
                if len(numeric_cols) >= 1:                    
                    for num_col in numeric_cols:
                        plt.figure(figsize=(10, 6))
                        sns.boxplot(x=df[kpi], 
                                    y=df[num_col], 
                                    hue = df[kpi], 
                                    legend = False,
                                    order=df.groupby(kpi, observed=False)[num_col].median().sort_values(ascending=False).index
                                    )
                        plt.title(f"Box Plot of {num_col} by {kpi}")
                        plt.xlabel(kpi)
                        plt.ylabel(num_col)
                        if df[kpi].nunique() > 10:
                            plt.xticks(rotation=270)
                        path = os.path.abspath(os.path.join(imgdir_path, f"Boxplot_{num_col}_vs_{kpi}.png"))
                        plt.savefig(path, bbox_inches='tight')
                        plt.close();
                        boxplot_data = self.get_boxplot_data(df, num_col, kpi)
                        print(boxplot_data)
                        boxplot_report = f"""Box Plot data of {num_col} by {kpi}\n\n{boxplot_data}\n\n"""
                        visualizations.append((f"Box Plot of {num_col} by {kpi}", path, boxplot_report))
            
                # Stacked Bar Chart for Categorical KPI against Categorical Columns
                if len(categorical_cols) >= 2:
                    for cat_col in categorical_cols:
                        if cat_col != kpi:
                            count = df[cat_col].nunique()
                            sorter = df[kpi].value_counts().index[-1]
                            tab = pd.crosstab(df[cat_col], df[kpi], normalize="index").sort_values(by=sorter, ascending=False)
                            plt.figure(figsize=(count + 5, 6))
                            tab.plot(kind='bar', stacked=True, colormap='Spectral')
                            plt.title(f"Stacked Bar Chart of {cat_col} by {kpi}")
                            plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
                            plt.xlabel(cat_col)
                            plt.ylabel(kpi)
                            if df[cat_col].nunique() > 10:
                                plt.xticks(rotation=270)
                            path = os.path.abspath(os.path.join(imgdir_path, f"Stacked_Bar_{kpi}_vs_{cat_col}.png"))
                            plt.savefig(path, bbox_inches='tight')
                            plt.close();
                            
                            stacked_bar_data = tab.to_markdown()
                            stacked_bar_report = f"Stacked Bar Chart data of {kpi} by {cat_col}\n\n{stacked_bar_data}\n\n"
                            visualizations.append((f"Stacked Bar Chart of {kpi} by {cat_col}", path, stacked_bar_report))

            # Scatter Plots for Numerical KPI against Numeric Columns
            if is_numeric_dtype(df[kpi]):
                if len(numeric_cols) >= 2:
                    for num_col in numeric_cols:
                        if num_col != kpi:
                            plt.figure(figsize=(10, 6))
                            sns.scatterplot(x=df[num_col], y=df[kpi], palette='Spectral')
                            plt.title(f"Scatter Plot of {kpi} vs {num_col}")
                            plt.xlabel(num_col)
                            plt.ylabel(kpi)
                            path = os.path.abspath(os.path.join(imgdir_path, f"Scatter_{kpi}_vs_{num_col}.png"))
                            plt.savefig(path, bbox_inches='tight')
                            plt.close();
                            
                            scatter_data = df[[kpi, num_col]].corr().to_markdown()
                            scatter_report = f"Scatter Plot data of {kpi} vs {num_col}\n\n{scatter_data}\n\n"
                            visualizations.append((f"Scatter Plot of {kpi} vs {num_col}", path, scatter_report))

                if len(categorical_cols) >= 1:
                    for cat_col in categorical_cols:
                        plt.figure(figsize=(10, 6))
                        sns.boxplot(x=df[cat_col], y=df[kpi], palette='Spectral', hue = df[cat_col], legend=False,
                                    order=df.groupby(cat_col, observed=False)[kpi].median().sort_values(ascending=False).index)
                        plt.title(f"Box Plot of {kpi} by {cat_col}")
                        plt.xlabel(cat_col)
                        plt.ylabel(kpi)
                        if df[cat_col].nunique() > 10:
                            plt.xticks(rotation=270)
                        path = os.path.abspath(os.path.join(imgdir_path, f"Boxplot_{kpi}_vs_{cat_col}.png"))
                        plt.savefig(path, bbox_inches='tight')
                        plt.close();
                        
                        boxplot_data = self.get_boxplot_data(df, kpi, cat_col)
                        boxplot_report = f"Box Plot data of {kpi} by {cat_col}\n\n{boxplot_data}\n\n"
                        visualizations.append((f"Box Plot of {kpi} by {cat_col}", path, boxplot_report))

        except Exception as e:
            print({e})
            raise Exception(f"An error occurred while generating bivariate plots: {e}")
        

        return visualizations


    def run(self, state:State):
        """
        Generates a report based on bivariate visualizations of the DataFrame.
        
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
            
            kpi = state.get("target_metric", None)
            
            # Call univariate analysis
            visualizations = self.bivariate(df, str(file_path), kpi)

            data_dictionary = state.get("data_dictionary", "")



            # Setup prompt
            prompt_correlation = """
            You are an expert data analyst. Analyze the following data {chart_data} and provide a concise report.

            **Leverage the {data_dictionary} context to incorporate any domain-specific insights.**

            The data is used to correlation heatmap.
            
            Your analysis should be concise and cover the following aspects:
            - Discuss the perceived strength and direction (positive/negative) of the correlation between key pairs of variables.
            - Highlight any pairs exhibiting strong direct or inverse relationships.
            - Point out any variables that show little to no apparent relationship.
            - Insights on what these relationships might imply about the connections between the variables.
            - Discuss the implications of the observed patterns in the context of the data.

            Please structure your response as a Markdown report with clear headings and bullet points.
            """

            prompt_boxplot = """
             You are an expert data analyst. Analyze the following data {chart_data} and provide a concise report.

            **Leverage the {data_dictionary} context to incorporate any domain-specific insights.**

            The data is used to create bivariate box plots.
            Your analysis should be concise and cover the following aspects:
            - Compare the distributions of the numeric variable across different categories:
                - Discuss  differences in central tendencies (50%) between groups
                - Discuss how the spread (75% minus 50%) varies between groups
                - Describe any apparent relationships between the categorical variable and the numeric variable.
            - Provide insights on how the categorical variable might influence the numeric variable.

            Please structure your response as a Markdown report with clear headings and bullet points.
            """

            prompt_stackedbarplot = """
             You are an expert data analyst. Analyze the following data {chart_data} and provide a concise report.

            **Leverage the {data_dictionary} context to incorporate any domain-specific insights.**

            The data is used to create stacked bar plots.
            Your analysis should be concise and cover the following aspects:
            - Compare the distributions of the categories within each stack
                - Discuss the relative contributions of each category to the total within each stack.
                - Identify the category that has the largest contribution in each stack
                - Discuss how the composition of the stacks changes across different groups.
            - Provide insights on how the categorical variables influence the composition of the stacks.

            Please structure your response as a Markdown report with clear headings and bullet points.
            """

            prompt_scatterplot = """
             You are an expert data analyst. Analyze the following data {chart_data} and provide a concise report.

            **Leverage the {data_dictionary} context to incorporate any domain-specific insights.**

            The data is used to create scatter plots.
            Your analysis should be concise and cover the following aspects:
            - Describe the relationship between the two numeric variables
                - Discuss the direction of the relationship (positive, negative, or no correlation).
                - Describe the strength of the relationship (strong, moderate, or weak).
            - Provide insights on how one variable might influence the other

            Please structure your response as a Markdown report with clear headings and bullet points.
            """

            # Analyze each visualization using the vision_analysis function
            report_content = "## Bivariate Analysis Report\n\n"
        
            for title, img_path, data in visualizations:
                encoded_image = encode_image_to_base64(img_path)
                image_markdown_tag = f"![{title}](data:image/png;base64,{encoded_image})"
                report_content += f"### {title}\n\n{image_markdown_tag}\n\n"
                if "Correlation" in title:
                    prompt_text = prompt_correlation.format(chart_data=data, data_dictionary=data_dictionary)
                elif "Box Plot" in title:
                    prompt_text = prompt_boxplot.format(chart_data=data, data_dictionary=data_dictionary)
                elif "Stacked Bar Chart" in title:
                    prompt_text = prompt_stackedbarplot.format(chart_data=data, data_dictionary=data_dictionary)
                elif "Scatter Plot" in title:
                    prompt_text = prompt_scatterplot.format(chart_data=data, data_dictionary=data_dictionary)
                response = self.llm.invoke(prompt_text)
                analysis_results = response.content if response else "No content returned from LLM."
                report_content += f"{analysis_results}"
        
            state["bivariate_analysis_report"] = report_content
            
        except Exception as e:
            state["error_message"].append(f"Error in bivariate analysis: {str(e)}\n\n")
            raise e

        return state
