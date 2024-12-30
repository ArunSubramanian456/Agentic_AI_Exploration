
import typer
from typing import Optional, List
from phi.agent import Agent
from phi.model.groq import Groq
from phi.storage.agent.postgres import PgAgentStorage
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.pgvector import PgVector, SearchType
from phi.embedder.sentence_transformer import SentenceTransformerEmbedder

import os
from dotenv import load_dotenv

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")


# STEP 1 - Setup a knowledge base to extract text from a PDF file and store it in a PostgreSQL database
# We will use PgVector which is an extension of PostgreSQL that allows us to store and search for vectors in the database
# We will use Dockers container to run the PostgreSQL database with pgvector extension
# The following command can be run in Git Bash to initialize the PostgreSQL database with pgvector extension within a Docker container
# Make sure to have Docker Desktop installed on your local machine before running the command

# docker run -d \
#   -e POSTGRES_DB=ai \
#   -e POSTGRES_USER=ai \
#   -e POSTGRES_PASSWORD=ai \
#   -e PGDATA=/var/lib/postgresql/data/pgdata \
#   -v pgvolume:/var/lib/postgresql/data \
#   -p 5532:5432 \
#   --name pgvector \
#   phidata/pgvector:16

# Once the command is run, check the Docker Desktop to see if the container "pgvector" 
# is running and the volume "pgvolume" is mounted to it

# Now we will create a PostgreSQL database connection string to connect to the database
# The connection string will contain the username, password, host, port, and database name
db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"


# Next, we will create a knowledge base to extract text from a PDF file and store it in the PostgreSQL database
# By default, phidata uses openai for embedding ; We will instead use SentenceTransformerEmbedder for embedding

knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db=PgVector(table_name="recipes", db_url=db_url, search_type=SearchType.hybrid),
    embedder=SentenceTransformerEmbedder(model_name="paraphrase-mpnet-base-v2"),
)

# We will now load the knowledge base to extract text from the PDF file and store it in the PostgreSQL database
knowledge_base.load(recreate=False, upsert=False)

# STEP 2 - Setup persistent storage for the agent by storing them in a database. 
# This will help agents remember the chat history
# We will use PgAgentStorage which is an extension of PostgreSQL that allows us to 
# store and search data related to agents in the database.
# This time we will use a different table name "pdf_agent" to store the data.

storage = PgAgentStorage(table_name = "pdf_agent", db_url=db_url)   

# STEP 3 - Create an agent to interact with the user and provide information from the knowledge base
# We will use the Agent class from the phi library to create an agent that can interact with the user

def pdf_agent(new: bool = False, user: str = "user"):
    session_id: Optional[str] = None

    if not new:
        existing_sessions: List[str] = storage.get_all_session_ids(user)
        if len(existing_sessions) > 0:
            session_id = existing_sessions[0]

    agent = Agent(
        session_id=session_id,
        user_id=user,
        knowledge=knowledge_base,
        storage=storage,
        model = Groq(id = "llama-3.1-70b-versatile", api_key=groq_api_key),
        add_context=True,
        # Show tool calls in the response
        show_tool_calls=True,
         # Enable Agents to use the knowledge base
        search_knowledge=True,
        # Enable the agent to read the chat history
        read_chat_history=True,
    )
    if session_id is None:
        session_id = agent.session_id
        print(f"Started Session: {session_id}\n")
    else:
        print(f"Continuing Session: {session_id}\n")

    # Runs the agent as a cli app
    agent.cli_app(markdown=True)


if __name__ == "__main__":
    # STEP 4 - Run the agent using Typer as a command-line interface
    typer.run(pdf_agent)

    # Finally, clean up when you are done
    # 1. Use CTRL + C to stop the agent interaction through CLI
    # 2. Deactivate the virtual environment by running "conda deactivate" in the terminal
    # 3. Stop the Docker container "pgvector" using Docker Desktop