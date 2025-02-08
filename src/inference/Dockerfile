FROM python:3.10

WORKDIR /app

COPY data/processed/test_lemma_tfidf.csv /app/data/processed/
COPY data/raw/test.csv /app/data/raw/
COPY outputs/models/Logistic_Regression_lemma_tfidf.pkl /app/outputs/models/

COPY src/inference/run_inference.py /app/inference/

COPY src/inference/requirements_inference.txt /app/inference/

RUN pip install --no-cache-dir -r /app/inference/requirements_inference.txt

CMD ["python3", "/app/inference/run_inference.py"]
