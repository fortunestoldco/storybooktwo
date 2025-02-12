from setuptools import setup, find_packages

setup(
    name="storybooktwo",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langchain>=0.1.0",
        "langchain-core>=0.1.0",
        "langchain-openai>=0.0.5",
        "langchain-community>=0.0.13",
        "langgraph>=0.0.20",
        "typing-extensions>=4.9.0",
        "tavily-python>=0.3.1",
        "langsmith>=0.0.87",
        "pymongo>=4.6.1",
        "python-dotenv>=1.0.1",
        "uvicorn>=0.27.0",
        "beautifulsoup4>=4.12.3",
        "requests>=2.31.0"
    ],
    python_requires=">=3.8",
)
