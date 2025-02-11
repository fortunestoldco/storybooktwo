FROM langchain/langgraph-api:3.11



ADD . /deps/storybook

RUN PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir -c /api/constraints.txt -e /deps/*
ENV LANGSERVE_GRAPHS='{"agent": "/deps/storybook/src/team/graph.py:graph"}'

WORKDIR /deps/storybook