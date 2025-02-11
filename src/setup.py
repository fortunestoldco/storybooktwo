from setuptools import setup, find_packages

setup(
    name="team",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "langchain",
        "langchain-core",
        "langchain-openai",
        "langchain-community",
        "langchain-experimental",
        "langgraph",
        "typing-extensions",
        "tavily-python",
    ],
    python_requires=">=3.8",
)
