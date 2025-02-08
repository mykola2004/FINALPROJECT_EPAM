FROM python:3.10

WORKDIR /app

COPY data /app/data
COPY models /app/models
COPY src/train /app/train

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python3", "/app/train/train.py"]