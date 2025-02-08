import os
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer
import logging

# Setting Up Loggings
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Identifying Customized NLTK Data Path
NLTK_DIR = "/root/nltk_data"
os.makedirs(NLTK_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DIR)

# Download Necessary NLTK Data
nltk.download('punkt', download_dir=NLTK_DIR)
nltk.download('stopwords', download_dir=NLTK_DIR)
nltk.download('wordnet', download_dir=NLTK_DIR)

# BASE Directory Setting
# For Docker use 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Train and Test Paths
train_data_path = os.path.join(BASE_DIR, "data", "raw", "train.csv")
test_data_path = os.path.join(BASE_DIR, "data", "raw", "test.csv")

if not os.path.exists(train_data_path):
    raise FileNotFoundError(f"Train data not found: {train_data_path}")
if not os.path.exists(test_data_path):
    raise FileNotFoundError(f"Test data not found: {test_data_path}")

# Load Data
data_train = pd.read_csv(train_data_path)
data_test = pd.read_csv(test_data_path)

logging.info("Start of preprocessing....")

# Drop duplicates
logging.info("Dropping duplicates....")
data_train = data_train.drop_duplicates()
data_test = data_test.drop_duplicates()

logging.info("Eliminating intersection between train and test sets....")
# Drop mutual reviews
duplicate_reviews = set(data_train["review"]) & set(data_test["review"])
data_train = data_train[~data_train["review"].isin(duplicate_reviews)]

# Clean reviews
logging.info("Deleting unnecessary characters....")
def clean_review(review_text):
    review_text = str(review_text)
    review_text = re.sub(r'http\S+', '', review_text)
    review_text = re.sub(r'\d+', '', review_text)
    review_text = re.sub(r'\W+', ' ', review_text)
    review_text = re.sub(r'\s+', ' ', review_text)
    return review_text.lower().strip()

data_train["review"] = data_train["review"].apply(clean_review)
data_test["review"] = data_test["review"].apply(clean_review)

# Tokenization
logging.info("Tokenizing reviews....")
data_train["review"] = data_train["review"].apply(word_tokenize)
data_test["review"] = data_test["review"].apply(word_tokenize)

# Stopword Removal
logging.info("Deleting stop words from reviews....")
stop_words = set(stopwords.words("english"))
def remove_stopwords(tokenized_text):
    return [word for word in tokenized_text if word not in stop_words]

data_train["review"] = data_train["review"].apply(remove_stopwords)
data_test["review"] = data_test["review"].apply(remove_stopwords)

# Lemmatization
logging.info("Lemmatizing reviews....")
lemmatizer = WordNetLemmatizer()
def apply_lemmatization(tokens):
    return [lemmatizer.lemmatize(word, wordnet.NOUN) for word in tokens]

data_train["LemmatizedReview"] = data_train["review"].apply(apply_lemmatization)
data_test["LemmatizedReview"] = data_test["review"].apply(apply_lemmatization)

# Stemmization 
logging.info("Stemmatization reviews....")
stemmer = PorterStemmer()
def apply_stemming(tokens):
    return [stemmer.stem(word) for word in tokens]
data_train['StemmatizedReview'] = data_train['review'].apply(apply_stemming)
data_test['StemmatizedReview'] = data_test['review'].apply(apply_stemming)

# Remove Short Words
logging.info("Deleting tokens of length <= 2....")
def remove_short_words(tokens):
    return [word for word in tokens if len(word) > 2]

data_train["LemmatizedReview"] = data_train["LemmatizedReview"].apply(remove_short_words)
data_test["LemmatizedReview"] = data_test["LemmatizedReview"].apply(remove_short_words)

data_train["StemmatizedReview"] = data_train["StemmatizedReview"].apply(remove_short_words)
data_test["StemmatizedReview"] = data_test["StemmatizedReview"].apply(remove_short_words)

# Convert Sentiment to Binary
logging.info("Preparing target feature....")
data_train["sentiment"] = data_train["sentiment"].map({"positive": 1, "negative": 0})
data_test["sentiment"] = data_test["sentiment"].map({"positive": 1, "negative": 0})

# Save Processed Data
train_processed_path = os.path.join(BASE_DIR, "data", "processed", "train.csv")
test_processed_path = os.path.join(BASE_DIR, "data", "processed", "test.csv")
os.makedirs(os.path.dirname(train_processed_path), exist_ok=True)

data_train.to_csv(train_processed_path, index=False)
data_test.to_csv(test_processed_path, index=False)

logging.info("Preprocessing completed. Cleaned data saved....")