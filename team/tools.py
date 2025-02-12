"""Tools for the creative writing system.

Current Date and Time (UTC): 2025-02-12 00:19:07
Current User's Login: fortunestoldco
"""

from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from typing import Dict, Any, List
import requests
from bs4 import BeautifulSoup
from pathlib import Path
import json

def tavily_tool(query: str) -> str:
    """Search the web using Tavily API."""
    search = TavilySearchAPIWrapper()
    results = search.run(query)
    return json.dumps(results, indent=2)

def scrape_webpages(urls: list) -> Dict[str, Any]:
    """Scrape content from provided URLs."""
    results = {}
    for url in urls:
        try:
            response = requests.get(
                url,
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            )
            soup = BeautifulSoup(response.text, 'html.parser')
            results[url] = {
                'title': soup.title.string if soup.title else '',
                'content': soup.get_text()[:5000]  # First 5000 chars
            }
        except Exception as e:
            results[url] = {'error': str(e)}
    return results

def write_document(content: str, filename: str) -> str:
    """Write content to a document."""
    filepath = Path(filename)
    filepath.write_text(content)
    return f"Written to {filepath}"

def read_document(filename: str) -> str:
    """Read content from a document."""
    filepath = Path(filename)
    if filepath.exists():
        return filepath.read_text()
    return f"File {filename} not found"

def edit_document(filename: str, content: str) -> str:
    """Edit an existing document."""
    filepath = Path(filename)
    if filepath.exists():
        filepath.write_text(content)
        return f"Updated {filepath}"
    return f"File {filename} not found"

def create_outline(title: str, points: List[str]) -> str:
    """Create a story outline."""
    outline = f"# {title}\n\n"
    for i, point in enumerate(points, 1):
        outline += f"{i}. {point}\n"
    return outline
