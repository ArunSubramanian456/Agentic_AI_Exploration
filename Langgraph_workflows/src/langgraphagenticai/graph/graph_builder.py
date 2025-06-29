from langgraph.graph import StateGraph, START,END
from langgraph.checkpoint.memory import MemorySaver
from src.langgraphagenticai.state.state import State
from src.langgraphagenticai.nodes.node_data_profiling import node_data_profiling
from src.langgraphagenticai.nodes.node_data_cleaning import node_data_cleaning
from src.langgraphagenticai.nodes.node_stats_summary import node_stats_summary
from src.langgraphagenticai.nodes.node_univariate_analysis import node_univariate_analysis
from src.langgraphagenticai.nodes.node_bivariate_analysis import node_bivariate_analysis
from src.langgraphagenticai.nodes.node_final_report import node_final_report
from langchain_core.runnables.graph import MermaidDrawMethod

class GraphBuilder:

    def __init__(self, model):
        self.llm=model
        self.graph_builder=StateGraph(State)
        self.memory = MemorySaver()


    def data_quality_graph(self):
        """
        Builds a data quality report graph using LangGraph.
        """
        self.node_data_profiling = node_data_profiling(self.llm)
        self.node_data_cleaning = node_data_cleaning(self.llm)
        self.node_stats_summary = node_stats_summary(self.llm)
        self.node_univariate_analysis = node_univariate_analysis(self.llm)
        self.node_bivariate_analysis = node_bivariate_analysis(self.llm)
        self.node_final_report = node_final_report(self.llm)


        # Add nodes for each step in the workflow (skip data_ingestion)
        self.graph_builder.add_node("data_profiling", self.node_data_profiling.run)
        self.graph_builder.add_node("data_cleaning", self.node_data_cleaning.run)
        self.graph_builder.add_node("stats_summary", self.node_stats_summary.run)
        self.graph_builder.add_node("univariate_analysis", self.node_univariate_analysis.run)
        self.graph_builder.add_node("bivariate_analysis", self.node_bivariate_analysis.run)
        self.graph_builder.add_node("final_step", self.node_final_report.run)

        # Add edges to define the flow (START -> data_profiling)
        self.graph_builder.add_edge(START, "data_profiling")
        self.graph_builder.add_edge("data_profiling", "data_cleaning")
        self.graph_builder.add_edge("data_cleaning", "stats_summary")
        self.graph_builder.add_edge("stats_summary", "univariate_analysis")
        self.graph_builder.add_edge("univariate_analysis", "bivariate_analysis")
        self.graph_builder.add_edge("bivariate_analysis", "final_step")
        self.graph_builder.add_edge("final_step", END)

    def setup_graph(self):
        """
        Sets up the graph
        """

        self.data_quality_graph()
        app = self.graph_builder.compile(
            interrupt_before=["data_profiling",
                              "data_cleaning", 
                              "stats_summary", 
                              "univariate_analysis", 
                              "bivariate_analysis", 
                              "final_step"],
            checkpointer = self.memory)
        
        # Save the graph image
        # self.save_graph_image(app)

        return app
    

    def save_graph_image(self,graph):
        # Generate the PNG image
        img_data = graph.get_graph().draw_mermaid_png(
            draw_method=MermaidDrawMethod.API
            )

        # Save the image to a file
        graph_path = "workflow_graph.png"
        with open(graph_path, "wb") as f:
            f.write(img_data)        
