"""Tool definitions for the hierarchical team agent."""

from typing import Annotated, Dict, List, Optional
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WebBaseLoader
from langchain_experimental.utilities import PythonREPL
from .configuration import WORKING_DIRECTORY

# Initialize tools
tavily_tool = TavilySearchResults(max_results=5)

@tool
def scrape_webpages(urls: List[str]) -> str:
    """Use requests and bs4 to scrape the provided web pages for detailed information."""
    loader = WebBaseLoader(urls)
    docs = loader.load()
    return "\n\n".join(
        [
            f'<Document name="{doc.metadata.get("title", "")}">\n{doc.page_content}\n</Document>'
            for doc in docs
        ]
    )

@tool
def create_outline(
    points: Annotated[List[str], "List of main points or sections."],
    file_name: Annotated[str, "File path to save the outline."],
    start: Annotated[Optional[int], "The start line. Default is 0"] = None,
    plot_points: Annotated[Optional[List[str]], "List of plot points."] = None,
    ending: Annotated[Optional[str], "The ending of the outline."] = None,
) -> Annotated[str, "Path of the saved outline file."]:
    """Create and save an outline."""
    with (WORKING_DIRECTORY / file_name).open("w") as file:
        if start is None:
            start = 0
        for i, point in enumerate(points[start:], start=start):
            file.write(f"{i + 1}. {point}\n")
        if plot_points:
            file.write("\nPlot Points:\n")
            for i, plot_point in enumerate(plot_points):
                file.write(f"{i + 1}. {plot_point}\n")
        if ending:
            file.write(f"\nEnding:\n{ending}\n")
    return f"Outline saved to {file_name}"

@tool
def read_document(
    file_name: Annotated[str, "File path to read the document from."],
    start: Annotated[Optional[int], "The start line. Default is 0"] = None,
    end: Annotated[Optional[int], "The end line. Default is None"] = None,
) -> str:
    """Read the specified document."""
    with (WORKING_DIRECTORY / file_name).open("r") as file:
        lines = file.readlines()
    if start is None:
        start = 0
    return "\n".join(lines[start:end])

@tool
def write_document(
    content: Annotated[str, "Text content to be written into the document."],
    file_name: Annotated[str, "File path to save the document."],
    start: Annotated[Optional[int], "The start line. Default is 0"] = None,
    plot_points: Annotated[Optional[List[str]], "List of plot points."] = None,
    ending: Annotated[Optional[str], "The ending of the document."] = None,
) -> Annotated[str, "Path of the saved document file."]:
    """Create and save a text document."""
    with (WORKING_DIRECTORY / file_name).open("w") as file:
        if start is None:
            start = 0
        file.write(content)
        if plot_points:
            file.write("\nPlot Points:\n")
            for i, plot_point in enumerate(plot_points):
                file.write(f"{i + 1}. {plot_point}\n")
        if ending:
            file.write(f"\nEnding:\n{ending}\n")
    return f"Document saved to {file_name}"

@tool
def edit_document(
    file_name: Annotated[str, "Path of the document to be edited."],
    inserts: Annotated[
        Dict[int, str],
        "Dictionary where key is the line number (1-indexed) and value is the text to be inserted at that line.",
    ],
) -> Annotated[str, "Path of the edited document file."]:
    """Edit a document by inserting text at specific line numbers."""
    with (WORKING_DIRECTORY / file_name).open("r") as file:
        lines = file.readlines()

    sorted_inserts = sorted(inserts.items())
    for line_number, text in sorted_inserts:
        if 1 <= line_number <= len(lines) + 1:
            lines.insert(line_number - 1, text + "\n")
        else:
            return f"Error: Line number {line_number} is out of range."

    with (WORKING_DIRECTORY / file_name).open("w") as file:
        file.writelines(lines)
    return f"Document edited and saved to {file_name}"

# Create Python REPL tool
repl = PythonREPL()

@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    return f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
