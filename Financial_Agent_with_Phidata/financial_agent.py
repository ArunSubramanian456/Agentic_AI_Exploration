from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo

import os
from dotenv import load_dotenv

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Create a web search agent
web_search_agent = Agent(name = "Web search agent",  
                         role = "Search the web for information",
                         model = Groq(id = "llama3-groq-70b-8192-tool-use-preview" ),
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

# Create a Multi AI Agentic Crew
multi_agent_crew = Agent(team = [web_search_agent, financial_agent],
                         model = Groq(id = "llama-3.1-70b-versatile"),
                         instructions=["Always include data sources", "Use tables to display the data"],
                         show_tool_calls=True,
                         markdown=True,
                        )

# Kickoff the crew
multi_agent_crew.print_response("Summarize analyst recommendations, current stock price, and stock fundamentals for AMZN", stream=True)