import openai
from phi.agent import Agent
import phi.api
from phi.model.openai import OpenAIChat
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
from phi.model.groq import Groq

import os
import phi
from phi.playground import Playground, serve_playground_app

#load environment variables
load_dotenv()
phi.api = os.getenv("PHI_API_KEY")

# Create a web search agent
web_search_agent = Agent(name = "Web search agent",  
                         role = "Search the web for information",
                         model = Groq(id = "llama3-groq-70b-8192-tool-use-preview"),
                         tools = [DuckDuckGo()],
                         instructions=["Always include data sources"],
                         show_tools_calls = True,
                         markdown = True,
)
                     
# Create a Financial agent
financial_agent = Agent(name = "Finance AI agent",
                        role = "Use the Yahoo Finance tool to gather finance data",
                        model = Groq(id = "llama3-groq-70b-8192-tool-use-preview"),
                        tools = [YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_news=True)],
                        instructions= ["Use tables to display the data"],
                        show_tool_calls=True,
                        markdown=True,
                        )

# Create Playground
app = Playground(agents=[web_search_agent, financial_agent], 
                 ).get_app()

if __name__ == '__main__':
    serve_playground_app("playground:app", reload=True)