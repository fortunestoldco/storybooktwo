#!/usr/bin/env python
"""Server runner for the hierarchical team agent.

Current Date and Time (UTC): 2025-02-11 23:49:46
Current User's Login: fortunestoldco
"""

from dotenv import load_dotenv
from pathlib import Path
import uvicorn
from langgraph_api import GraphApplication

# Load environment variables from project root
load_dotenv()

# Initialize the LangGraph application
app = GraphApplication()

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=2024)
