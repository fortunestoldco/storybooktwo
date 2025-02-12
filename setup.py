from setuptools import setup, find_packages

setup(
    name="storybooktwo",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langchain",
        "langchain-core",
        "langchain-openai",
        "langchain-community",
        "langchain-experimental",
        "langgraph",
        "typing-extensions",
        "tavily-python",
        "langsmith",
        "pymongo",
        "python-dotenv",
        "uvicorn"
    ],
    python_requires=">=3.8",
)
