"""Server runner for the creative writing system.

Current Date and Time (UTC): 2025-02-12 00:38:05
Current User's Login: fortunestoldco
"""

from dotenv import load_dotenv
from pathlib import Path
import uvicorn
from langgraph_api import GraphApplication
from team.graph import graph

# Load environment variables
load_dotenv()

# Initialize the LangGraph application
app = GraphApplication()

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=2024)