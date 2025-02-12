"""Tools for the hierarchical team agent.

Current Date and Time (UTC): 2025-02-12 01:39:53
Current User's Login: fortunestoldco
"""

import os
import json
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
from langchain_community.tools import BaseTool
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from bs4 import BeautifulSoup
import requests
import matplotlib.pyplot as plt
import seaborn as sns

class FileSystemTool(BaseTool):
    """Base class for file system tools."""
    working_dir: Path
    
    def __init__(self, working_dir: Path):
        super().__init__()
        self.working_dir = working_dir
        self.working_dir.mkdir(parents=True, exist_ok=True)

class DocumentReader(FileSystemTool):
    """Tool for reading documents."""
    name = "read_document"
    description = "Read the contents of a document file"
    
    def _run(self, filename: str) -> str:
        filepath = self.working_dir / filename
        if not filepath.exists():
            return f"Error: File {filename} does not exist"
        return filepath.read_text()

class DocumentWriter(FileSystemTool):
    """Tool for writing documents."""
    name = "write_document"
    description = "Write content to a document file"
    
    def _run(self, filename: str, content: str) -> str:
        filepath = self.working_dir / filename
        filepath.write_text(content)
        return f"Successfully wrote to {filename}"

class DocumentEditor(FileSystemTool):
    """Tool for editing documents."""
    name = "edit_document"
    description = "Edit an existing document file"
    
    def _run(self, filename: str, content: str) -> str:
        filepath = self.working_dir / filename
        if not filepath.exists():
            return f"Error: File {filename} does not exist"
        filepath.write_text(content)
        return f"Successfully updated {filename}"

class OutlineCreator(FileSystemTool):
    """Tool for creating document outlines."""
    name = "create_outline"
    description = "Create a document outline and save it"
    
    def _run(self, filename: str, sections: List[str]) -> str:
        filepath = self.working_dir / filename
        content = "\n".join(f"# {section}" for section in sections)
        filepath.write_text(content)
        return f"Successfully created outline in {filename}"

class WebScraper(BaseTool):
    """Tool for web scraping."""
    name = "scrape_webpages"
    description = "Scrape content from web pages"
    
    def _run(self, urls: List[str]) -> Dict[str, str]:
        results = {}
        for url in urls:
            try:
                response = requests.get(url)
                soup = BeautifulSoup(response.text, 'html.parser')
                results[url] = soup.get_text()
            except Exception as e:
                results[url] = f"Error: {str(e)}"
        return results

class ChartGenerator(FileSystemTool):
    """Tool for generating charts using Python."""
    name = "python_repl_tool"
    description = "Generate charts and save them as images"
    
    def _run(self, code: str) -> str:
        try:
            # Create a new figure for each chart
            plt.figure()
            
            # Execute the plotting code
            exec(code, {'plt': plt, 'sns': sns})
            
            # Save the figure
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chart_{timestamp}.png"
            filepath = self.working_dir / filename
            plt.savefig(filepath)
            plt.close()
            
            return f"Successfully created chart: {filename}"
        except Exception as e:
            return f"Error generating chart: {str(e)}"

def create_tools(config):
    """Create tool instances with configuration."""
    return {
        "read_document": DocumentReader(config.working_directory),
        "write_document": DocumentWriter(config.working_directory),
        "edit_document": DocumentEditor(config.working_directory),
        "create_outline": OutlineCreator(config.working_directory),
        "scrape_webpages": WebScraper(),
        "python_repl_tool": ChartGenerator(config.working_directory),
        "tavily_tool": TavilySearchAPIWrapper(tavily_api_key=config.tavily_api_key)
    }
