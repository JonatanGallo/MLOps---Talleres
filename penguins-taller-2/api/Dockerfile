FROM python:3.9-slim

RUN pip install uv

WORKDIR /app

COPY . /app

RUN uv sync --frozen

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8989"]