FROM python:3.11

copy . .

RUN pip3 install --no-cache-dir -r requirements.txt

RUN python Train.py

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8989"]
