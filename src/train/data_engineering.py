import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import logging

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()])

# For Docker Use
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def vectorizing_method(train, test, vectorizer, train_output_path, test_output_path, column):
    X_train = vectorizer.fit_transform(train[column].apply(lambda x: " ".join(eval(x))))
    X_test = vectorizer.transform(test[column].apply(lambda x: " ".join(eval(x))))
    
    train_engineered = pd.DataFrame(X_train.toarray(), columns=vectorizer.get_feature_names_out())
    test_engineered = pd.DataFrame(X_test.toarray(), columns=vectorizer.get_feature_names_out())
    
    train_engineered["sentiment"] = train["sentiment"]
    test_engineered["sentiment"] = test["sentiment"]

    train_engineered.to_csv(train_output_path, index=False)
    test_engineered.to_csv(test_output_path, index=False)

def vectorize_data(train_input, test_input):
    train_data = pd.read_csv(train_input)
    test_data = pd.read_csv(test_input)

    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
    count_vectorizer = CountVectorizer(ngram_range=(1,2), max_features=5000)

    #TF-IDF lemma data engineering
    logging.info("Vectorizing(TF-IDF) dataset(lemmatized)....")
    train_output_path = os.path.join(BASE_DIR, "data", "processed", "train_lemma_tfidf.csv")
    test_output_path = os.path.join(BASE_DIR, "data", "processed", "test_lemma_tfidf.csv")
    column = "LemmatizedReview"
    vectorizing_method(train_data, test_data, tfidf_vectorizer, train_output_path, test_output_path, column)
    logging.info(f"Vectorized(TF-IDF) dataset(lemmatized) data saved to {train_output_path} and {test_output_path}....")

    #TF-IDF stemma data engineering
    logging.info("Vectorizing(TF-IDF) dataset(stemming)....")
    train_output_path = os.path.join(BASE_DIR, "data", "processed", "train_stem_tfidf.csv")
    test_output_path = os.path.join(BASE_DIR, "data", "processed", "test_stem_tfidf.csv")
    column = "StemmatizedReview"
    vectorizing_method(train_data, test_data, tfidf_vectorizer, train_output_path, test_output_path, column)
    logging.info(f"Vectorized(TF-IDF) dataset(stemmatized) data saved to {train_output_path} and {test_output_path}....")

    #Count lemma data engineering
    logging.info("Vectorizing(Count) dataset(lemmatized)....")
    train_output_path = os.path.join(BASE_DIR, "data", "processed", "train_lemma_count.csv")
    test_output_path = os.path.join(BASE_DIR, "data", "processed", "test_lemma_count.csv")
    column = "LemmatizedReview"
    vectorizing_method(train_data, test_data, count_vectorizer, train_output_path, test_output_path, column)
    logging.info(f"Vectorized(Count) dataset(lemmatized) data saved to {train_output_path} and {test_output_path}....")

    #Count stemma data engineering
    logging.info("Vectorizing(Count) dataset(stemming)....")
    train_output_path = os.path.join(BASE_DIR, "data", "processed", "train_stem_count.csv")
    test_output_path = os.path.join(BASE_DIR, "data", "processed", "test_stem_count.csv")
    column = "StemmatizedReview"
    vectorizing_method(train_data, test_data, count_vectorizer, train_output_path, test_output_path, column)
    logging.info(f"Vectorized(Count) dataset(stemmatized) data saved to {train_output_path} and {test_output_path}....")
    
train_input = os.path.join(BASE_DIR, "data", "processed", "train.csv")
test_input = os.path.join(BASE_DIR, "data", "processed", "test.csv")
    
vectorize_data(train_input, test_input)