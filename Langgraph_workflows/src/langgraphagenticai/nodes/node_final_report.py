import nltk
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os
from langchain.chains.summarize import load_summarize_chain
from src.langgraphagenticai.state.state import State
from langchain_core.prompts import PromptTemplate

class node_final_report:
    def __init__(self,  model):
        """
        Initializes the node for final report.
        This node is responsible for generating the final report.
        """
        self.llm = model

        try:
            nltk.data.find('tokenizers/punkt')
        except Exception as e:
            print("Downloading NLTK 'punkt' tokenizer data...")
            nltk.download('punkt')

        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except Exception as e:
            print("Downloading NLTK 'averaged_perceptron_tagger' data...")
            nltk.download('averaged_perceptron_tagger')


    def run(self, state:State):
        """
        Runs the node to generate final report
        """
        try:
            univariate_analysis_report = state.get("univariate_analysis_report", None)
            bivariate_analysis_report = state.get("bivariate_analysis_report", None)

            final_report = f"""{univariate_analysis_report}\n\n{bivariate_analysis_report}\n\n"""

             # Get file path from state and read the file
            file_path = state.get("data_file_path", None)
            if file_path is None:
                raise ValueError("File path is required for profiling.")   
            
            parentdir_path = os.path.dirname(file_path)
            finalreport_path = os.path.abspath(os.path.join(parentdir_path, "final_report.md"))

            with open(finalreport_path, "w", encoding="utf-8") as f:
                f.write(final_report)
            
            # Load markdown file
            loader = UnstructuredMarkdownLoader(finalreport_path)
            data = loader.load()

            #  Apply chunking
            final_documents=RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=100).split_documents(data)
            final_documents

            chunks_prompt="""
            You are an expert data analyst. Review the analysis report and summarize the findings.
                    
            Here is the analysis report to review:
            {text}

            Summary:
            """
            map_prompt_template=PromptTemplate(input_variables=['text'],
                                                template=chunks_prompt)
            
            final_prompt='''
            You are an expert data analyst. Review the analysis report and share your conclusions and recommendations to improve the business.

            Your output should have
            - Conclusions based on the data analysis
            - Recommendations to the business

            Structure your output in markdown format.

            Here are the analysis report to review:
            {text}

            '''
            final_prompt_template=PromptTemplate(input_variables=['text'],template=final_prompt)


            summary_chain=load_summarize_chain(
            llm=self.llm,
            chain_type="map_reduce",
            map_prompt=map_prompt_template,
            combine_prompt=final_prompt_template,
            verbose=True
            )

            output=summary_chain.invoke(final_documents)

            state["final_report"] = output['output_text']

            
        except Exception as e:
            state["error_message"].append(f"Error in final report: {str(e)}\n\n")
            raise e

        return state
