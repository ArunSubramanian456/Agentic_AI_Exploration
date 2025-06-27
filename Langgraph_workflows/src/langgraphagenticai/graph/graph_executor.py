from src.langgraphagenticai.state.state import State, make_initial_state
import uuid
import src.langgraphagenticai.utils.constants as const
from langgraph.graph import StateGraph, START,END

class GraphExecutor:
    def __init__(self, graph):
        self.graph = graph

    def get_config(self, thread_id):
        return {"configurable": {"thread_id": thread_id}}
    
    def initialize_graph(self, data_file_path):
        """
        Initiates the graph and store data file path in State
        """

        graph = self.graph

        # Setup thread_id for initialization
        thread_id = str(uuid.uuid4())
        config = self.get_config(thread_id)

        # resume the graph
        state = make_initial_state()
        state["data_file_path"] = data_file_path
        
        for output in self.graph.stream(state, config, stream_mode="values"):
            state = output
        
        return {"thread_id" : thread_id, "eda_state" : state}
    

    def graph_execution(self, thread_id, state, stage):
        """
        Executes the data profile report workflow for a given thread ID and State.
        """
        if stage == const.PROFILE_DATA:
            state.next_node=const.CLEAN_DATA
            execute_as_node = START

        if stage == const.CLEAN_DATA:
            state.next_node=const.SUMMARIZE_DATA
            execute_as_node = const.PROFILE_DATA

        if stage == const.SUMMARIZE_DATA:
            state.next_node=const.GENERATE_UNIVARIATE_REPORT
            execute_as_node = const.CLEAN_DATA

        if stage == const.GENERATE_UNIVARIATE_REPORT:
            state.next_node=const.GENERATE_BIVARIATE_REPORT
            execute_as_node = const.SUMMARIZE_DATA

        if stage == const.GENERATE_BIVARIATE_REPORT:
            state.next_node=const.GENERATE_FINAL_REPORT
            execute_as_node = const.GENERATE_UNIVARIATE_REPORT

        if stage == const.GENERATE_FINAL_REPORT:
            state.next_node=const.END_NODE
            execute_as_node = const.GENERATE_BIVARIATE_REPORT

        return self.update_and_resume_graph(state, thread_id, as_node=execute_as_node)
    
    
    ## -------- Helper Method to handle the graph resume state ------- ##

    def update_and_resume_graph(self, state, thread_id, as_node):
        graph = self.graph
        thread = self.get_config(thread_id)
        
        graph.update_state(thread, state, as_node=as_node)
        
        # Resume the graph
        state = None
        for output in graph.stream(None, thread, stream_mode="values"):
            state = output
        
        return {"thread_id" : thread_id, "eda_state" : state}