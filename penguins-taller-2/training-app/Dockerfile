FROM jupyter/datascience-notebook:latest

RUN pip install uv

WORKDIR /app

COPY . /app

RUN uv sync --frozen

CMD ["uv", "run", "jupyter", "lab"]